import os
import time
import random
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_CSV = "NormAd_train.csv"

DECOMPOSED_CACHE = "normad_decomposed.csv"  # trigger/target/spec per row
TRIGGER_EMBED_CACHE = "normad_trigger_embs.npy"
TARGET_EMBED_CACHE = "normad_target_embs.npy"
SPEC_EMBED_CACHE = "normad_spec_embs.npy"
PREMISE_CACHE = "normad_premises_v2.csv"
PAIRS_CHECKPOINT = "normad_pairs_v2_ckpt.csv"  # checkpoint

PAIRS_PER_ANCHOR = 7

# Grouping thresholds (trigger + target)
TRIGGER_SIM_THRESHOLD = 0.75
TARGET_SIM_THRESHOLD = 0.75

# Weakener threshold: spec_sim must be BELOW this to be considered opposite
SPEC_SPLIT_THRESHOLD = 0.70

# GPT verification of weakeners
VERIFY_WEAKENERS = True

GPT_MODEL = "gpt-4.1-nano"  # used for decomposition and premise generation
VERIFY_MODEL = "gpt-4.1"  # stronger model used only for weakener verification
EMBED_MODEL = "text-embedding-3-small"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ==========================================
# 2. HELPERS
# ==========================================
def gpt_call(system, user, temperature=0.0, max_tokens=150, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            time.sleep(0.35)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [GPT error attempt {attempt+1}]: {e}")
            time.sleep(2**attempt + random.uniform(0, 1))
    return None


def embed_texts(texts, desc="Embedding", batch_size=100):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend(item.embedding for item in resp.data)
        time.sleep(0.3)
    return np.array(all_embeddings, dtype=np.float32)


def load_or_build_emb_cache(texts, path, n, desc):
    if os.path.exists(path):
        arr = np.load(path)
        if arr.shape[0] == n:
            print(f"  Loaded {desc} embeddings: {arr.shape}")
            return arr
        print(f"  [WARN] Cache size mismatch for {desc}. Rebuilding...")
        os.remove(path)
    print(f"  Building {desc} embeddings...")
    arr = embed_texts(texts, desc=desc)
    np.save(path, arr)
    print(f"  Saved {desc} embeddings: {arr.shape}")
    return arr


# ==========================================
# 3. DATA LOADING
# ==========================================
print("Step 1: Loading NormAd...")
df = pd.read_csv(INPUT_CSV)
df = df.reset_index(drop=True)
df["row_id"] = df.index.astype(int)

for col in ["Country", "Rule-of-Thumb", "Value", "Explanation", "Subaxis", "Story"]:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: str(x).replace("###", "").strip())
for col in ["Explanation", "Story"]:
    if col in df.columns:
        df[col] = df[col].replace("None", pd.NA)

print(
    f"  {len(df)} rows | {df['Subaxis'].nunique()} subaxes | "
    f"{df['Country'].nunique()} countries"
)


# ==========================================
# 4. DECOMPOSITION  (cached)
# ==========================================
DECOMPOSE_SYSTEM = """\
You decompose cultural behavior rules into exactly 3 components.
Reply ONLY with valid JSON — no markdown, no explanation.

{
  "trigger": "the specific situation or context when the rule applies (2-6 words, lowercase)",
  "target": "the exact aspect of behavior being regulated (2-5 words, lowercase)",
  "specification": "what is prescribed or forbidden, starting with a verb (3-8 words, lowercase)"
}

Examples:
Rule: "One should use both hands when giving a gift"
{"trigger": "giving a gift", "target": "how to present the gift", "specification": "use both hands"}

Rule: "It is correct to avoid opening gifts when they are received"
{"trigger": "receiving a gift", "target": "when to open the gift", "specification": "do not open the gift immediately"}

Rule: "Arriving slightly late to social gatherings is acceptable"
{"trigger": "arriving at social gatherings", "target": "when to arrive", "specification": "arriving slightly late is acceptable"}

Rule: "It is polite to wait until everyone has been served before starting to eat"
{"trigger": "sitting down to a meal", "target": "when to start eating", "specification": "wait until everyone is served"}\
"""


def decompose_rule(rule, explanation=""):
    expl = (
        explanation
        if pd.notna(explanation) and str(explanation).strip() not in ("", "None")
        else ""
    )
    user = f"Rule: {rule}" + (f"\nContext: {expl}" if expl else "")
    raw = gpt_call(DECOMPOSE_SYSTEM, user, temperature=0.0, max_tokens=120)
    if raw is None:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text.strip())
        if all(k in parsed for k in ("trigger", "target", "specification")):
            return {k: str(v).strip().lower() for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass
    return None


def load_or_build_decomposed(df):
    if os.path.exists(DECOMPOSED_CACHE):
        print(f"Step 2: Loading cached decompositions from '{DECOMPOSED_CACHE}'...")
        cached = pd.read_csv(DECOMPOSED_CACHE)
        df = df.merge(
            cached[["row_id", "trigger", "target", "specification"]],
            on="row_id",
            how="left",
        )
        ok = df["trigger"].notna().sum()
        print(f"  Loaded {ok} / {len(df)} decompositions.")
    else:
        print(f"Step 2: Decomposing {len(df)} rules via GPT (one-time)...")
        triggers, targets, specs = [], [], []
        failed = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Decomposing"):
            result = decompose_rule(row["Rule-of-Thumb"], row.get("Explanation", ""))
            if result:
                triggers.append(result["trigger"])
                targets.append(result["target"])
                specs.append(result["specification"])
            else:
                triggers.append(None)
                targets.append(None)
                specs.append(None)
                failed += 1
        df = df.copy()
        df["trigger"] = triggers
        df["target"] = targets
        df["specification"] = specs
        df[["row_id", "trigger", "target", "specification"]].to_csv(
            DECOMPOSED_CACHE, index=False
        )
        print(f"  Done. {failed} failures. Saved to '{DECOMPOSED_CACHE}'.")
    return df


# ==========================================
# 5. EMBEDDINGS  (cached)
# ==========================================
def load_or_build_all_embeddings(df_valid):
    print("Step 4: Loading/building embeddings...")
    trigger_embs = load_or_build_emb_cache(
        df_valid["trigger"].tolist(), TRIGGER_EMBED_CACHE, len(df_valid), "trigger"
    )
    target_embs = load_or_build_emb_cache(
        df_valid["target"].tolist(), TARGET_EMBED_CACHE, len(df_valid), "target"
    )
    spec_embs = load_or_build_emb_cache(
        df_valid["specification"].tolist(),
        SPEC_EMBED_CACHE,
        len(df_valid),
        "specification",
    )
    return trigger_embs, target_embs, spec_embs


# ==========================================
# 6. PREMISE PRE-COMPUTATION
# ==========================================
PREMISE_SYSTEM = """\
Write a single neutral scene-setting sentence for a cultural norm inference task.

Given a cultural rule and the situation it applies to, describe a realistic scenario
where the rule would be relevant — WITHOUT saying what the character does or whether
their behavior is correct.

Requirements:
- One sentence only, no more
- Use a generic name (e.g. Alex, Sam, Jordan)
- Describe the situation/context only — stop before the character acts
- Do not reveal whether the behavior is correct or incorrect
- Do not mention any country or culture
- End with a comma or 'and' to leave the action open, OR end just before the action

Examples:

Rule: "It is polite to receive and give items with both hands to show respect"
Trigger: "giving or receiving items"
Premise: At a friend's gathering, Alex was about to hand a small gift to the host.

Rule: "It is expected to initially decline offers before accepting after insistence"
Trigger: "being offered food or drink by a host"
Premise: During a visit to a friend's house, Sam was offered a second serving of dessert by the host.

Rule: "It is correct to arrive at the exact scheduled time for any appointment"
Trigger: "arriving at social gatherings"
Premise: Jordan had been invited to a dinner party scheduled to begin at 7 PM.

Rule: "It is respectful to wait for the host to signal before starting to eat"
Trigger: "sitting down to a meal"
Premise: At a dinner party, Alex sat down at the table as the host finished serving everyone.

Output ONLY the single premise sentence, nothing else.\
"""


def rule_to_premise(rule, trigger):
    """Generate a neutral scene-setting premise from the rule and its trigger."""
    user = f"Rule: {rule}\nTrigger: {trigger}"
    result = gpt_call(PREMISE_SYSTEM, user, temperature=0.4, max_tokens=80)
    return result.strip() if result else ""


def load_or_build_premises(df_valid):
    """Compute one premise per valid row, cache to CSV."""
    if os.path.exists(PREMISE_CACHE):
        print(f"Step 5: Loading cached premises from '{PREMISE_CACHE}'...")
        cached = pd.read_csv(PREMISE_CACHE)
        merged = df_valid[["row_id"]].merge(cached, on="row_id", how="left")
        already_done = merged["premise"].notna().sum()
        print(f"  Loaded {already_done} / {len(df_valid)} premises.")

        missing_mask = merged["premise"].isna()
        if missing_mask.any():
            print(f"  Computing {missing_mask.sum()} missing premises...")
            for pos in tqdm(merged[missing_mask].index, desc="Premises"):
                row = df_valid.loc[pos]
                merged.at[pos, "premise"] = rule_to_premise(
                    row["Rule-of-Thumb"], row.get("trigger", "")
                )
            merged[["row_id", "premise"]].to_csv(PREMISE_CACHE, index=False)

        df_valid = df_valid.copy()
        df_valid["premise"] = merged["premise"].fillna("").values
    else:
        print(f"Step 5: Computing premises for {len(df_valid)} rows (one-time)...")
        premises = []
        for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Premises"):
            premises.append(
                rule_to_premise(row["Rule-of-Thumb"], row.get("trigger", ""))
            )
        df_valid = df_valid.copy()
        df_valid["premise"] = premises
        df_valid[["row_id", "premise"]].to_csv(PREMISE_CACHE, index=False)
        print(f"  Saved premises to '{PREMISE_CACHE}'.")
    return df_valid


# ==========================================
# 7. WEAKENER VERIFICATION
# ==========================================
VERIFY_SYSTEM = """\
You are checking whether two cultural rules prescribe GENUINELY OPPOSITE behaviors
in the same situation.

Answer YES only if:
- Both rules address the same specific action in the same context
- One prescribes doing X and the other prescribes NOT doing X (or doing the opposite)
- Following one rule would mean VIOLATING the other

Answer NO if:
- The rules are paraphrases or logically equivalent (same prescription, different wording)
- One rule is a STRICTER version of the other (e.g. "always use both hands" vs
  "use both hands or right hand only" — the strict version satisfies the lenient one)
- Both rules could reasonably be followed at the same time
- The difference is about degree or style, not direction
- The rules address different aspects of the same situation

Reply with ONLY 'YES' or 'NO'.\
"""


def verify_weakener(anchor_rule, candidate_rule):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=VERIFY_MODEL,
                temperature=0.0,
                max_tokens=5,
                messages=[
                    {"role": "system", "content": VERIFY_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Rule A: {anchor_rule}\nRule B: {candidate_rule}",
                    },
                ],
            )
            time.sleep(0.35)
            result = resp.choices[0].message.content.strip()
            return result.upper().startswith("Y")
        except Exception as e:
            print(f"  [verify error attempt {attempt+1}]: {e}")
            time.sleep(2**attempt + random.uniform(0, 1))
    return True  # fail open on API error


# ==========================================
# 8. 4-ROW ENTRY BUILDER  (NEW)
#
#  Given a verified anchor-match pair, emit 4 rows:
#
#  1. hypothesis=anchor_rule, premise=anchor_premise,
#     update=anchor_culture  -> strengthener
#  2. hypothesis=anchor_rule, premise=anchor_premise,
#     update=match_culture   -> weakener
#  3. hypothesis=match_rule,  premise=match_premise,
#     update=match_culture   -> strengthener
#  4. hypothesis=match_rule,  premise=match_premise,
#     update=anchor_culture  -> weakener
# ==========================================
def build_four_rows(anchor_row, match_row, trigger_sim, target_sim, spec_sim):
    anchor_rule = str(anchor_row["Rule-of-Thumb"]).strip()
    match_rule = str(match_row["Rule-of-Thumb"]).strip()
    anchor_country = str(anchor_row["Country"]).strip()
    match_country = str(match_row["Country"]).strip()
    anchor_premise = str(anchor_row.get("premise", "")).strip()
    match_premise = str(match_row.get("premise", "")).strip()
    topic = str(anchor_row.get("Subaxis", "")).strip()

    shared_meta = {
        "topic": topic,
        "anchor_trigger": str(anchor_row.get("trigger", "")),
        "anchor_target": str(anchor_row.get("target", "")),
        "anchor_specification": str(anchor_row.get("specification", "")),
        "match_trigger": str(match_row.get("trigger", "")),
        "match_target": str(match_row.get("target", "")),
        "match_specification": str(match_row.get("specification", "")),
        "anchor_rule": anchor_rule,
        "match_rule": match_rule,
        "trigger_sim": round(float(trigger_sim), 4),
        "target_sim": round(float(target_sim), 4),
        "spec_sim": round(float(spec_sim), 4),
        "normad_anchor_idx": int(anchor_row.name),
        "normad_match_idx": int(match_row.name),
    }

    rows = [
        # Row 1: anchor is its own strengthener
        {
            "premise": anchor_premise,
            "hypothesis": anchor_rule,
            "update": f"They are a {anchor_country} national.",
            "label": "strengthener",
            "country/cultural group": anchor_country,
            **shared_meta,
        },
        # Row 2: anchor rule, match culture -> weakener
        {
            "premise": anchor_premise,
            "hypothesis": anchor_rule,
            "update": f"They are a {match_country} national.",
            "label": "weakener",
            "country/cultural group": match_country,
            **shared_meta,
        },
        # Row 3: match rule is its own strengthener
        {
            "premise": match_premise,
            "hypothesis": match_rule,
            "update": f"They are a {match_country} national.",
            "label": "strengthener",
            "country/cultural group": match_country,
            **shared_meta,
        },
        # Row 4: match rule, anchor culture -> weakener
        {
            "premise": match_premise,
            "hypothesis": match_rule,
            "update": f"They are a {anchor_country} national.",
            "label": "weakener",
            "country/cultural group": anchor_country,
            **shared_meta,
        },
    ]
    return rows


# ==========================================
# 9. ANCHOR SAMPLING  (stratified by Subaxis)
# ==========================================
def sample_anchors_stratified(df, seed=RANDOM_SEED):
    rng = random.Random(seed)
    selected = []
    for _, grp in df.groupby("Subaxis"):
        idxs = grp.index.tolist()
        rng.shuffle(idxs)
        selected.extend(idxs)
    return selected


# ==========================================
# 10. MAIN PIPELINE
# ==========================================
if __name__ == "__main__":

    # Step 2: decompose (cached)
    df = load_or_build_decomposed(df)

    # Step 3: filter to rows with valid decompositions
    df_valid = df[
        df["trigger"].notna() & df["target"].notna() & df["specification"].notna()
    ].copy()
    df_valid = df_valid.reset_index(drop=True)
    print(f"\nStep 3: Valid decomposed rows: {len(df_valid)} / {len(df)}")

    # Step 4: embeddings (cached)
    trigger_embs, target_embs, spec_embs = load_or_build_all_embeddings(df_valid)
    idx_to_pos = {idx: pos for pos, idx in enumerate(df_valid.index)}

    # Step 5: premises (NEW — one per valid row, cached)
    df_valid = load_or_build_premises(df_valid)

    # Step 6: anchors (all valid rows, stratified)
    anchor_indices = sample_anchors_stratified(df_valid)
    print(f"\nStep 6: {len(anchor_indices)} potential anchors.")

    # Step 7: resume from checkpoint
    if os.path.exists(PAIRS_CHECKPOINT):
        print(f"\nStep 7: Resuming from '{PAIRS_CHECKPOINT}'...")
        ckpt = pd.read_csv(PAIRS_CHECKPOINT)
        all_rows = ckpt.to_dict("records")
        # A verified pair is identified by (anchor_idx, match_idx) — both directions covered
        done_pairs = {(r["normad_anchor_idx"], r["normad_match_idx"]) for r in all_rows}
        done_anchors = {r["normad_anchor_idx"] for r in all_rows}
        print(f"  Loaded {len(all_rows)} rows | {len(done_anchors)} anchors done.")
    else:
        all_rows, done_pairs, done_anchors = [], set(), set()

    skipped = 0
    weakeners_rejected = 0
    pairs_found = 0
    total_anchors = len(anchor_indices)

    print(f"\nBuilding pairs...\n")

    for iteration, anchor_idx in enumerate(anchor_indices):
        anchor_row = df_valid.loc[anchor_idx]
        anchor_country = anchor_row["Country"]
        anchor_subaxis = anchor_row["Subaxis"]
        anchor_rule = anchor_row["Rule-of-Thumb"]
        anchor_pos = idx_to_pos[anchor_idx]

        if anchor_idx in done_anchors:
            continue

        # -------------------------------------------------------
        # GROUPING: same subaxis, different country,
        # trigger sim >= threshold AND target sim >= threshold
        # -------------------------------------------------------
        same_subaxis = df_valid[
            (df_valid["Subaxis"] == anchor_subaxis)
            & (df_valid.index != anchor_idx)
            & (df_valid["Country"] != anchor_country)
        ]

        if len(same_subaxis) < 1:
            skipped += 1
            continue

        cand_positions = [idx_to_pos[i] for i in same_subaxis.index]

        trig_sims = cosine_similarity(
            trigger_embs[anchor_pos].reshape(1, -1), trigger_embs[cand_positions]
        )[0]
        tgt_sims = cosine_similarity(
            target_embs[anchor_pos].reshape(1, -1), target_embs[cand_positions]
        )[0]

        group_mask = (trig_sims >= TRIGGER_SIM_THRESHOLD) & (
            tgt_sims >= TARGET_SIM_THRESHOLD
        )
        grouped = same_subaxis[group_mask].copy()
        grouped["_trig_sim"] = trig_sims[group_mask]
        grouped["_tgt_sim"] = tgt_sims[group_mask]

        if len(grouped) < 1:
            skipped += 1
            continue

        # -------------------------------------------------------
        # WEAKENER POOL: spec_sim below threshold
        # -------------------------------------------------------
        grp_positions = [idx_to_pos[i] for i in grouped.index]
        spec_sims = cosine_similarity(
            spec_embs[anchor_pos].reshape(1, -1), spec_embs[grp_positions]
        )[0]
        grouped["_spec_sim"] = spec_sims

        raw_weakener_pool = grouped[
            grouped["_spec_sim"] <= SPEC_SPLIT_THRESHOLD
        ].sort_values("_spec_sim", ascending=True)

        if len(raw_weakener_pool) == 0:
            skipped += 1
            continue

        # -------------------------------------------------------
        # WEAKENER VERIFICATION
        # -------------------------------------------------------
        if VERIFY_WEAKENERS:
            verified = []
            for _, w_row in raw_weakener_pool.iterrows():
                pair_key = (
                    min(anchor_idx, int(w_row.name)),
                    max(anchor_idx, int(w_row.name)),
                )
                if pair_key in done_pairs:
                    continue
                if verify_weakener(anchor_rule, w_row["Rule-of-Thumb"]):
                    verified.append(w_row)
                else:
                    weakeners_rejected += 1
            weakener_pool = pd.DataFrame(verified) if verified else pd.DataFrame()
        else:
            weakener_pool = raw_weakener_pool

        if len(weakener_pool) == 0:
            skipped += 1
            continue

        # -------------------------------------------------------
        # EMIT 4 ROWS PER VERIFIED PAIR
        # Cap at PAIRS_PER_ANCHOR verified pairs per anchor
        # -------------------------------------------------------
        pairs_this_anchor = 0

        for _, w_row in weakener_pool.iterrows():
            if pairs_this_anchor >= PAIRS_PER_ANCHOR:
                break

            match_idx = int(w_row.name)
            pair_key = (min(anchor_idx, match_idx), max(anchor_idx, match_idx))

            # Skip if this pair was already processed from the other direction
            if pair_key in done_pairs:
                continue

            four_rows = build_four_rows(
                anchor_row=anchor_row,
                match_row=df_valid.loc[match_idx],
                trigger_sim=float(w_row["_trig_sim"]),
                target_sim=float(w_row["_tgt_sim"]),
                spec_sim=float(w_row["_spec_sim"]),
            )

            all_rows.extend(four_rows)
            done_pairs.add(pair_key)
            done_anchors.add(anchor_idx)
            done_anchors.add(match_idx)  # match is also "done" as an anchor
            pairs_this_anchor += 1
            pairs_found += 1

            # Save immediately on first pair so the file appears on disk right away
            if pairs_found == 1:
                pd.DataFrame(all_rows).to_csv(PAIRS_CHECKPOINT, index=False)

        if pairs_this_anchor == 0:
            skipped += 1
        else:
            print(
                f"  [{iteration+1:04d}/{total_anchors}] {anchor_country} | {anchor_subaxis} | "
                f"pairs={pairs_this_anchor} | total_rows={len(all_rows)}"
            )

        # Checkpoint every 20 anchors — always write so the file exists even
        # when most anchors are skipped, giving visible progress on disk
        if iteration % 20 == 0:
            pd.DataFrame(
                all_rows
                if all_rows
                else [{"status": f"processing... iteration {iteration}"}]
            ).to_csv(PAIRS_CHECKPOINT, index=False)

    # Final checkpoint flush
    if all_rows:
        pd.DataFrame(all_rows).to_csv(PAIRS_CHECKPOINT, index=False)

    # ==========================================
    # 11. FINAL OUTPUT
    # ==========================================
    final_df = pd.DataFrame(all_rows)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"  Total rows          : {len(final_df)}")
    print(f"  Verified pairs      : {pairs_found}")
    print(f"  Weakeners rejected  : {weakeners_rejected}")
    print(f"  Skipped anchors     : {skipped}")
    if len(final_df):
        print(
            f"  Label distribution  :\n{final_df['label'].value_counts().to_string()}"
        )
    print("=" * 70)

    if len(final_df) > 0:
        pd.set_option("display.max_colwidth", 100)
        print("\nSample output (first 8 rows):")
        for _, row in final_df.head(8).iterrows():
            print(
                f"\n  [{row['label'].upper()}] "
                f"{row['country/cultural group']} / {row['topic']}"
            )
            print(f"  Premise   : {str(row['premise'])[:90]}")
            print(f"  Hypothesis: {row['hypothesis'][:90]}")
            print(f"  Update    : {row['update']}")
            print(
                f"  Sims      : trig={row['trigger_sim']} "
                f"tgt={row['target_sim']} spec={row['spec_sim']}"
            )

        print(f"\nspec_sim per label:")
        print(
            final_df.groupby("label")["spec_sim"]
            .agg(["mean", "min", "max"])
            .round(3)
            .to_string()
        )
        print(f"\nTop topics:")
        print(final_df["topic"].value_counts().to_string())

        final_df.to_csv("normad_pairs_v2.csv", index=False)
        final_df.to_excel("normad_pairs_v2.xlsx", index=False)
        print("\nSaved: normad_pairs_v2.csv and normad_pairs_v2.xlsx")
    else:
        print("\n[WARNING] No pairs generated.")
        print("Try lowering thresholds:")
        print(f"  TRIGGER_SIM_THRESHOLD = {TRIGGER_SIM_THRESHOLD} -> try 0.65")
        print(f"  TARGET_SIM_THRESHOLD  = {TARGET_SIM_THRESHOLD}  -> try 0.65")
        print(f"  SPEC_SPLIT_THRESHOLD  = {SPEC_SPLIT_THRESHOLD}  -> try 0.60")
        print("  VERIFY_WEAKENERS = False  -> see raw volume without GPT filter")
