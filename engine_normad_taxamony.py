import os
import time
import random
import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_ID = "akhilayerukola/NormAd"

NORMALIZED_CACHE = "normad_normalized.csv"
TAXONOMY_CACHE = "normad_taxonomy.json"
CLASSIFIED_CACHE = "normad_classified.csv"
PAIRS_CHECKPOINT = "normad_pairs_checkpoint.csv"

N_ANCHORS = 120
PAIRS_PER_ANCHOR = 7

# Taxonomy construction
TAXONOMY_SAMPLE_SIZE = 300  # rows sampled to generate candidate labels
TARGET_TAXONOMY_SIZE = 50  # number of categories

GPT_MODEL = "gpt-4o-mini"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ==========================================
# 2. GPT HELPER
# ==========================================
def gpt_call(
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int = 200,
    retries: int = 3,
) -> str | None:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            result = response.choices[0].message.content.strip()
            time.sleep(0.5)  # proactive TPM pacing — prevents rate limit errors
            return result
        except Exception as e:
            print(f"  [GPT error attempt {attempt+1}]: {e}")
            time.sleep(2**attempt + random.uniform(0, 1))
    return None


# ==========================================
# 3. DATA LOADING & CLEANING
# ==========================================
print("Step 1: Loading NormAd...")
dataset = load_dataset(DATASET_ID)
split_name = list(dataset.keys())[0]
df = pd.DataFrame(dataset[split_name])

# Add a stable row_id that survives any reordering
df = df.reset_index(drop=True)
df["row_id"] = df.index.astype(int)

for col in [
    "Country",
    "Rule-of-Thumb",
    "Value",
    "Explanation",
    "Subaxis",
    "Story",
    "Background",
]:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: str(x).replace("###", "").strip())

for col in ["Explanation", "Story", "Background"]:
    if col in df.columns:
        df[col] = df[col].replace("None", pd.NA)

print(
    f"  Loaded {len(df)} rows | "
    f"null explanations: {df['Explanation'].isna().sum()} | "
    f"null stories: {df['Story'].isna().sum()}"
)


# ==========================================
# 4. RULE NORMALIZATION (cached, keyed by row_id)
# ==========================================
NORMALIZE_SYSTEM = (
    "You are helping build a cross-cultural NLP dataset. "
    "Rewrite a cultural rule into a single, explicit, stance-clear declarative sentence.\n"
    "Rules:\n"
    "- Always start with 'One should' or 'One should not'\n"
    "- Make the concrete physical or social ACTION explicit\n"
    "- If conditional, resolve to the dominant expected behavior\n"
    "- If the rule is about NOT doing something, use 'One should not'\n"
    "- Remove culture-specific references, proper nouns, and vague language\n"
    "- Output ONLY the rewritten sentence, nothing else\n\n"
    "Examples:\n"
    "Input: 'Follow the host's cue regarding footwear upon entering their home.'\n"
    "Output: 'One should remove shoes before entering someone's home.'\n\n"
    "Input: 'Wearing shoes indoors is considered a sign of hospitality.'\n"
    "Output: 'One should keep shoes on when entering someone's home.'\n\n"
    "Input: 'It is impolite to point your feet toward another person.'\n"
    "Output: 'One should not point feet toward another person.'"
)


def normalize_rule(rule: str, value: str, explanation: str = "") -> str | None:
    user = (
        f"Rule: {rule}\n"
        f"Underlying value: {value}\n"
        f"Explanation: {explanation if pd.notna(explanation) and str(explanation).strip() not in ('', 'None') else 'Not provided'}\n\n"
        f"Rewrite:"
    )
    return gpt_call(NORMALIZE_SYSTEM, user, temperature=0.2, max_tokens=80)


def load_or_build_normalized_df(df: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(NORMALIZED_CACHE):
        print(f"Step 2: Loading cached normalized rules from '{NORMALIZED_CACHE}'...")
        cached = pd.read_csv(NORMALIZED_CACHE)
        if "row_id" in cached.columns:
            df = df.merge(
                cached[["row_id", "normalized_rule"]], on="row_id", how="left"
            )
        else:
            if len(cached) == len(df):
                print(
                    f"  [WARN] Old cache format (no row_id) — aligning by position. "
                    f"Delete '{NORMALIZED_CACHE}' to regenerate with stable keys."
                )
                df = df.copy()
                df["normalized_rule"] = cached["normalized_rule"].values
            else:
                raise ValueError(
                    f"Old cache has {len(cached)} rows but dataset has {len(df)} rows. "
                    f"Delete '{NORMALIZED_CACHE}' and rerun to rebuild it."
                )
        loaded = df["normalized_rule"].notna().sum()
        print(f"  Loaded {loaded} normalized rules.")
    else:
        print(
            f"Step 2: Normalizing {len(df)} rules via GPT (one-time, will be cached)..."
        )
        normalized_rules = []
        failed = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Normalizing rules"):
            result = normalize_rule(
                row["Rule-of-Thumb"], row.get("Value", ""), row.get("Explanation", "")
            )
            if result is None:
                result = row["Rule-of-Thumb"]
                failed += 1
            normalized_rules.append(result)

        df = df.copy()
        df["normalized_rule"] = normalized_rules
        df[["row_id", "normalized_rule"]].to_csv(NORMALIZED_CACHE, index=False)
        print(
            f"  Done. {failed} fallbacks to original rule. Saved to '{NORMALIZED_CACHE}'."
        )
    return df


# ==========================================
# 5. TAXONOMY GENERATION (cached)
#    Step A: sample rows → free-form candidate labels (1 per row)
#    Step B: GPT canonicalizes them into a fixed list of TARGET_TAXONOMY_SIZE labels
# ==========================================
CANDIDATE_SYSTEM = (
    "You are a cross-cultural NLP researcher. "
    "Given a cultural rule about behavior, output a short abstract category label (4-8 words) "
    "that captures WHAT behavior the rule is about — not whether it is good or bad.\n"
    "Examples:\n"
    "  Rule: 'One should remove shoes before entering someone's home.' "
    "→ 'footwear when entering homes'\n"
    "  Rule: 'One should not point feet toward another person.' "
    "→ 'foot positioning toward other people'\n"
    "  Rule: 'One should inform the host before bringing additional guests.' "
    "→ 'notifying host about additional guests'\n"
    "Output ONLY the category label, nothing else, all lowercase."
)


def generate_candidate_category(rule: str) -> str | None:
    result = gpt_call(CANDIDATE_SYSTEM, f"Rule: {rule}", temperature=0.3, max_tokens=25)
    return result.strip().lower() if result else None


def build_canonicalize_system(target_n: int) -> str:
    return (
        "You are organizing a cross-cultural NLP taxonomy.\n"
        "Below is a messy list of behavioral category labels generated from cultural rules.\n"
        f"Your job: merge near-duplicates and return EXACTLY {target_n} clean, distinct labels.\n\n"
        "Rules:\n"
        "- Each label: 4-8 words, all lowercase\n"
        "- Labels describe WHAT behavior, not whether it is good or bad\n"
        "- Merge semantically identical or near-identical labels into one\n"
        "- Keep labels broad enough that multiple cultural rules can fall under them\n"
        f"- Output EXACTLY {target_n} labels, one per line\n"
        "- No numbering, no bullet points, no extra text"
    )


def generate_taxonomy(df: pd.DataFrame) -> list[str]:
    sample_size = min(TAXONOMY_SAMPLE_SIZE, len(df))
    sample_rules = (
        df["normalized_rule"]
        .dropna()
        .sample(sample_size, random_state=RANDOM_SEED)
        .tolist()
    )

    print(f"  Generating {sample_size} candidate labels...")
    candidates = []
    for rule in tqdm(sample_rules, desc="Candidate labels"):
        cat = generate_candidate_category(rule)
        if cat:
            candidates.append(cat)

    unique_candidates = list(set(candidates))
    print(
        f"  Got {len(unique_candidates)} unique candidates. Canonicalizing to {TARGET_TAXONOMY_SIZE}..."
    )

    system = build_canonicalize_system(TARGET_TAXONOMY_SIZE)
    result = gpt_call(
        system,
        "Candidate labels:\n" + "\n".join(unique_candidates),
        temperature=0.2,
        max_tokens=TARGET_TAXONOMY_SIZE * 20,
    )
    if result is None:
        raise RuntimeError(
            "GPT failed to canonicalize taxonomy. Check your API key/quota."
        )

    taxonomy = [
        line.strip().lower() for line in result.strip().splitlines() if line.strip()
    ]
    if len(taxonomy) < TARGET_TAXONOMY_SIZE * 0.8:
        raise RuntimeError(
            f"Taxonomy too small ({len(taxonomy)} labels). "
            f"Expected ~{TARGET_TAXONOMY_SIZE}. Check GPT output."
        )
    print(f"  Canonical taxonomy: {len(taxonomy)} categories.")
    return taxonomy


def load_or_build_taxonomy(df: pd.DataFrame) -> list[str]:
    if os.path.exists(TAXONOMY_CACHE):
        print(f"Step 3: Loading cached taxonomy from '{TAXONOMY_CACHE}'...")
        with open(TAXONOMY_CACHE) as f:
            taxonomy = json.load(f)
        print(f"  Loaded {len(taxonomy)} categories.")
    else:
        print("Step 3: Building taxonomy (one-time)...")
        taxonomy = generate_taxonomy(df)
        with open(TAXONOMY_CACHE, "w") as f:
            json.dump(taxonomy, f, indent=2)
        print(f"  Saved taxonomy to '{TAXONOMY_CACHE}'.")
    return taxonomy


# ==========================================
# 6. ROW CLASSIFICATION (cached, keyed by row_id)
#    Each row gets one category from the fixed taxonomy.
# ==========================================
def build_classify_system(taxonomy: list[str]) -> str:
    taxonomy_str = "\n".join(f"- {cat}" for cat in taxonomy)
    return (
        "You are classifying cultural rules into behavioral categories.\n"
        "Pick the single most appropriate category from this fixed list:\n\n"
        f"{taxonomy_str}\n\n"
        "Rules:\n"
        "- Output ONLY the exact category label from the list — copy it character-for-character\n"
        "- No explanation, no punctuation changes, no modifications\n"
        "- If none fit perfectly, pick the closest one"
    )


def classify_rule(rule: str, classify_system: str, taxonomy_set: set) -> str | None:
    result = gpt_call(classify_system, f"Rule: {rule}", temperature=0.0, max_tokens=30)
    if result is None:
        return None
    result = result.strip().lower()
    if result in taxonomy_set:
        return result
    for cat in taxonomy_set:
        if result in cat or cat in result:
            return cat
    return None  # unclassifiable — filtered out later


def load_or_build_classified_df(df: pd.DataFrame, taxonomy: list[str]) -> pd.DataFrame:
    taxonomy_set = set(taxonomy)

    if os.path.exists(CLASSIFIED_CACHE):
        print(f"Step 4: Loading cached classifications from '{CLASSIFIED_CACHE}'...")
        cached = pd.read_csv(CLASSIFIED_CACHE)
        if "row_id" in cached.columns:
            df = df.merge(cached[["row_id", "category"]], on="row_id", how="left")
        else:
            if len(cached) == len(df):
                print(
                    f"  [WARN] Old cache format (no row_id) — aligning by position. "
                    f"Delete '{CLASSIFIED_CACHE}' to regenerate with stable keys."
                )
                df = df.copy()
                df["category"] = cached["category"].values
            else:
                raise ValueError(
                    f"Old cache has {len(cached)} rows but dataset has {len(df)} rows. "
                    f"Delete '{CLASSIFIED_CACHE}' and rerun to rebuild it."
                )
        in_taxonomy = df["category"].isin(taxonomy_set).sum()
        print(f"  Loaded {in_taxonomy} classified rows.")
    else:
        print(f"Step 4: Classifying {len(df)} rules (one-time, will be cached)...")
        classify_system = build_classify_system(taxonomy)
        categories = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
            cat = classify_rule(row["normalized_rule"], classify_system, taxonomy_set)
            categories.append(cat)

        df = df.copy()
        df["category"] = categories
        df[["row_id", "category"]].to_csv(CLASSIFIED_CACHE, index=False)

        in_taxonomy = df["category"].isin(taxonomy_set).sum()
        print(
            f"  Done. In-taxonomy: {in_taxonomy}/{len(df)}. Saved to '{CLASSIFIED_CACHE}'."
        )

    return df


# ==========================================
# 7. RELATIONAL LABEL: does candidate SUPPORT or CONTRADICT anchor?
# ==========================================
LABEL_SYSTEM = (
    "You are labeling pairs of cultural rules for an NLP dataset.\n\n"
    "Given an ANCHOR rule and a CANDIDATE rule:\n"
    "- Reply 'strengthener' if the candidate SUPPORTS or AGREES with the anchor "
    "(i.e. both rules recommend the same behavior)\n"
    "- Reply 'weakener' if the candidate CONTRADICTS or CONFLICTS with the anchor "
    "(i.e. the candidate recommends the opposite or incompatible behavior)\n"
    "- Reply 'unrelated' if the two rules are about fundamentally different behaviors "
    "and cannot meaningfully be compared as supporting or contradicting each other\n\n"
    "Important:\n"
    "- Focus on whether the underlying recommended BEHAVIOR is the same or opposite\n"
    "- 'One should not X' can still be a strengthener if it means the same as 'One should avoid X'\n"
    "- Do NOT be misled by surface negation — judge by semantic meaning\n"
    "- Use 'unrelated' when the rules address different topics entirely "
    "(e.g. one is about food preparation, the other about informing a host)\n"
    "- Reply ONLY with 'strengthener', 'weakener', or 'unrelated' — nothing else"
)


def get_relational_label(anchor_rule: str, candidate_rule: str) -> str | None:
    """
    Determine whether candidate_rule supports (strengthener), contradicts (weakener),
    or is unrelated to the anchor_rule.
    Returns 'strengthener', 'weakener', 'unrelated', or None on API failure.
    Callers should discard 'unrelated' pairs.
    """
    user = (
        f"Anchor rule: {anchor_rule}\n"
        f"Candidate rule: {candidate_rule}\n\n"
        f"Does the candidate support, contradict, or is it unrelated to the anchor?"
    )
    result = gpt_call(LABEL_SYSTEM, user, temperature=0.0, max_tokens=10)
    if result is None:
        return None
    result = result.strip().lower()
    if result in ("strengthener", "weakener", "unrelated"):
        return result
    # Handle slight variations
    if result.startswith("strength"):
        return "strengthener"
    if result.startswith("weak"):
        return "weakener"
    if result.startswith("unrel"):
        return "unrelated"
    return None


# ==========================================
# 8. ANCHOR SAMPLING (stratified by Subaxis)
# ==========================================
def sample_anchors_stratified(
    df: pd.DataFrame, n: int, seed: int = RANDOM_SEED
) -> list[int]:
    try:
        subaxis_groups = df.groupby("Subaxis").apply(
            lambda g: g.index.tolist(), include_groups=False
        )
    except TypeError:
        subaxis_groups = df.groupby("Subaxis").apply(lambda g: g.index.tolist())

    n_subaxes = len(subaxis_groups)
    base_per_subaxis = n // n_subaxes
    remainder = n % n_subaxes

    rng = random.Random(seed)
    selected = []
    for i, (_, idxs) in enumerate(subaxis_groups.items()):
        quota = base_per_subaxis + (1 if i < remainder else 0)
        picked = rng.sample(idxs, min(quota, len(idxs)))
        selected.extend(picked)

    if len(selected) < n:
        pool = [i for i in df.index.tolist() if i not in set(selected)]
        extra = rng.sample(pool, min(n - len(selected), len(pool)))
        selected.extend(extra)

    rng.shuffle(selected)
    return selected[:n]


# ==========================================
# 9. STORY → DECLARATIVE PREMISE
# ==========================================
STORY_TO_PREMISE_SYSTEM = (
    "You are editing sentences for an NLP dataset. "
    "Convert the given story (which ends with a yes/no question) into a declarative statement "
    "that describes what happened, without asking a question.\n"
    "Rules:\n"
    "- Remove the question at the end\n"
    "- Keep the scenario description intact\n"
    "- Replace any country name with 'this culture'\n"
    "- Output ONLY the declarative sentence, nothing else"
)


def story_to_premise(story: str, country: str) -> str | None:
    if pd.isna(story) or not str(story).strip():
        return None
    blinded = (
        str(story)
        .replace(country, "this culture")
        .replace(country.title(), "this culture")
    )
    return gpt_call(
        STORY_TO_PREMISE_SYSTEM, f"Story: {blinded}", temperature=0.2, max_tokens=120
    )


# ==========================================
# 10. PAIR ENTRY BUILDER
# ==========================================
def build_pair_entry(
    anchor_row: pd.Series,
    match_row: pd.Series,
    pair_label: str,
    premise: str,
    anchor_idx: int,
    match_idx: int,
) -> dict:
    match_country = str(match_row.get("Country", "")).strip()
    explanation = match_row.get("Explanation", "")
    if pd.isna(explanation) or str(explanation).strip() in ("", "None"):
        explanation = ""

    return {
        "premise": premise,
        "hypothesis": str(anchor_row.get("Rule-of-Thumb", "")).strip(),
        "update": f"They are a {match_country} national.",
        "label": pair_label,
        "country/cultural group": match_country,
        "topic": str(match_row.get("Subaxis", "")).strip(),
        "category": str(match_row.get("category", "")).strip(),
        "explanation": str(explanation).strip(),
        "match_rule_of_thumb": str(match_row.get("Rule-of-Thumb", "")).strip(),
        "match_normalized_rule": str(match_row.get("normalized_rule", "")).strip(),
        "anchor_normalized_rule": str(anchor_row.get("normalized_rule", "")).strip(),
        "normad_idx": int(match_idx),
        "anchor_idx": int(anchor_idx),
    }


# ==========================================
# 11. MAIN PIPELINE
# ==========================================
if __name__ == "__main__":

    # --- Steps 2-4: normalize, build taxonomy, classify (all cached after first run) ---
    df = load_or_build_normalized_df(df)
    taxonomy = load_or_build_taxonomy(df)
    taxonomy_set = set(taxonomy)
    df = load_or_build_classified_df(df, taxonomy)

    # --- Filter to rows we can actually use ---
    valid_mask = df["normalized_rule"].notna() & df["category"].isin(taxonomy_set)
    df_valid = df[valid_mask].copy()
    print(f"\nValid rows after filtering: {len(df_valid)} / {len(df)}")
    print(
        f"\nTop 20 categories:\n{df_valid['category'].value_counts().head(20).to_string()}"
    )

    # --- Step 5: sample anchors ---
    print(f"\nStep 5: Sampling {N_ANCHORS} anchors (stratified by Subaxis)...")
    anchor_indices = sample_anchors_stratified(df_valid, N_ANCHORS)
    print(f"  Sampled {len(anchor_indices)} anchors.")

    # --- Step 6: build pairs (resume from checkpoint if available) ---
    if os.path.exists(PAIRS_CHECKPOINT):
        print(f"Step 6: Resuming from checkpoint '{PAIRS_CHECKPOINT}'...")
        ckpt_df = pd.read_csv(PAIRS_CHECKPOINT)
        all_pairs = ckpt_df.to_dict("records")
        # Rebuild seen_pairs so we don't re-emit the same rows
        seen_pairs = set()
        for p in all_pairs:
            seen_pairs.add((p["anchor_idx"], p["normad_idx"], p["label"]))
        done_anchors = set(ckpt_df["anchor_idx"].unique())
        support_total = int((ckpt_df["label"] == "strengthener").sum())
        contrast_total = int((ckpt_df["label"] == "weakener").sum())
        print(
            f"  Loaded {len(all_pairs)} pairs | {len(done_anchors)} anchors already done."
        )
    else:
        all_pairs = []
        seen_pairs = set()
        done_anchors = set()
        support_total = 0
        contrast_total = 0

    skipped_anchors: list[int] = []

    print(f"\nStep 6: Building pairs across {N_ANCHORS} anchors...\n")

    for iteration, anchor_idx in enumerate(anchor_indices):
        anchor_row = df_valid.loc[anchor_idx]
        anchor_cat = anchor_row["category"]
        anchor_country = anchor_row["Country"]
        anchor_story = anchor_row.get("Story", pd.NA)
        anchor_norm_rule = anchor_row["normalized_rule"]

        # Skip anchors already completed in a previous run
        if anchor_idx in done_anchors:
            print(
                f"[{iteration+1:03d}/{N_ANCHORS}] idx={anchor_idx} — already done, skipping."
            )
            continue

        print(
            f"[{iteration+1:03d}/{N_ANCHORS}] idx={anchor_idx} | "
            f"{anchor_country} | {anchor_cat}"
        )
        print(f"  Anchor rule: {anchor_norm_rule[:100]}")

        # --- Find all candidate rows: same category, different country ---
        same_cat = df_valid[
            (df_valid["category"] == anchor_cat)
            & (df_valid.index != anchor_idx)
            & (df_valid["Country"] != anchor_country)
        ]

        if len(same_cat) < 2:
            print(
                f"  [SKIP] Not enough candidates in same category ({len(same_cat)}).\n"
            )
            skipped_anchors.append(anchor_idx)
            continue

        print(
            f"  Candidate pool: {len(same_cat)} rows from {same_cat['Country'].nunique()} countries"
        )

        # --- Convert story → declarative premise ---
        premise = story_to_premise(anchor_story, anchor_country)
        if premise is None:
            if pd.notna(anchor_story) and str(anchor_story).strip():
                premise = str(anchor_story).replace(anchor_country, "this culture")
                print(f"  [WARN] GPT premise failed — using blinded raw story.")
            else:
                premise = ""
                print(f"  [WARN] No story available — premise left blank.")
        else:
            print(f"  Premise: {premise[:100]}...")

        # --- Shuffle candidate pool for variety ---
        rng_local = random.Random(RANDOM_SEED + anchor_idx)
        candidates = (
            same_cat.sample(frac=1, random_state=rng_local.randint(0, 9999))
            .reset_index()  # df_valid index → column named 'index'
            .to_dict("records")
        )

        # --- Label each candidate relative to the anchor via GPT ---
        # We collect verified strengtheners and weakeners separately,
        # then pair them up. We stop once we have PAIRS_PER_ANCHOR of each
        # (or exhaust the candidate pool).
        verified_strengtheners: list[dict] = []
        verified_weakeners: list[dict] = []

        print(f"  Labeling candidates...")
        for cand in candidates:
            if (
                len(verified_strengtheners) >= PAIRS_PER_ANCHOR
                and len(verified_weakeners) >= PAIRS_PER_ANCHOR
            ):
                break  # have enough of both — no need to label more

            cand_orig_idx = int(cand["index"])
            cand_norm_rule = cand["normalized_rule"]

            label = get_relational_label(anchor_norm_rule, cand_norm_rule)
            if label is None or label == "unrelated":
                continue  # discard — unrelated rules or API failure

            cand["_label"] = label
            if (
                label == "strengthener"
                and len(verified_strengtheners) < PAIRS_PER_ANCHOR
            ):
                verified_strengtheners.append(cand)
            elif label == "weakener" and len(verified_weakeners) < PAIRS_PER_ANCHOR:
                verified_weakeners.append(cand)

        print(
            f"  Verified → {len(verified_strengtheners)} strengtheners | "
            f"{len(verified_weakeners)} weakeners"
        )

        if len(verified_strengtheners) == 0 or len(verified_weakeners) == 0:
            print(f"  [SKIP] Need at least 1 of each type.\n")
            skipped_anchors.append(anchor_idx)
            continue

        # --- Pair up: each strengthener gets exactly one weakener ---
        # Constraint: strengthener and weakener must be from different countries
        used_w_indices: set[int] = set()
        pairs_written = 0

        for s_rec in verified_strengtheners:
            if pairs_written >= PAIRS_PER_ANCHOR:
                break

            s_orig_idx = int(s_rec["index"])
            s_key = (anchor_idx, s_orig_idx, "strengthener")
            if s_key in seen_pairs:
                continue

            # Find the first unused weakener from a different country
            matched_w = None
            for w_rec in verified_weakeners:
                w_orig_idx = int(w_rec["index"])
                if w_orig_idx in used_w_indices:
                    continue
                w_key = (anchor_idx, w_orig_idx, "weakener")
                if w_key in seen_pairs:
                    continue
                if s_rec["Country"] == w_rec["Country"]:
                    continue
                matched_w = (w_orig_idx, w_key, w_rec)
                break

            if matched_w is None:
                continue

            w_orig_idx, w_key, w_rec = matched_w

            # Commit
            seen_pairs.add(s_key)
            seen_pairs.add(w_key)
            used_w_indices.add(w_orig_idx)

            all_pairs.append(
                build_pair_entry(
                    anchor_row,
                    df_valid.loc[s_orig_idx],
                    "strengthener",
                    premise,
                    anchor_idx,
                    s_orig_idx,
                )
            )
            all_pairs.append(
                build_pair_entry(
                    anchor_row,
                    df_valid.loc[w_orig_idx],
                    "weakener",
                    premise,
                    anchor_idx,
                    w_orig_idx,
                )
            )

            support_total += 1
            contrast_total += 1
            pairs_written += 1

        if pairs_written == 0:
            print(f"  [SKIP] Could not form any complete pairs.\n")
            skipped_anchors.append(anchor_idx)
        else:
            print(
                f"  Pairs written: {pairs_written} | "
                f"Running total: {support_total + contrast_total}\n"
            )

        # --- Checkpoint: save after every anchor so crashes lose nothing ---
        if all_pairs:
            pd.DataFrame(all_pairs).to_csv(PAIRS_CHECKPOINT, index=False)

    # ==========================================
    # 12. FINAL OUTPUT
    # ==========================================
    final_df = pd.DataFrame(all_pairs)

    print("=" * 80)
    print("PIPELINE COMPLETE")
    print(f"  Total rows collected : {len(final_df)}")
    print(f"  Strengtheners        : {support_total}")
    print(f"  Weakeners            : {contrast_total}")
    print(f"  Skipped anchors      : {len(skipped_anchors)}")
    if skipped_anchors:
        print(f"  Skipped indices      : {skipped_anchors}")
    print("=" * 80)

    if len(final_df) > 0:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_colwidth", 120)
        print("\nSample output (first 10 rows):")
        print(
            final_df[
                [
                    "label",
                    "country/cultural group",
                    "category",
                    "anchor_normalized_rule",
                    "match_normalized_rule",
                ]
            ]
            .head(10)
            .to_string()
        )

        final_df.to_csv("normad_pairs_v3.csv", index=False)
        final_df.to_excel("normad_pairs_v3.xlsx", index=False)
        print("\nSaved: normad_pairs_v3.csv and normad_pairs_v3.xlsx")
    else:
        print(
            "\n[WARNING] No pairs generated. "
            "Try lowering TARGET_TAXONOMY_SIZE (broader categories = bigger pools)."
        )
