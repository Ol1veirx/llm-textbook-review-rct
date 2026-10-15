"""Sentence-level look at false positives and false negatives.

Writes three files to results/error_examples/:
  sentence_level_counts.csv  one row per sentence, with how many participants flagged it
  false_positives_llm.csv    LLM suggestions on error-free sentences, with accept / reject counts
  false_negatives.csv        reference errors that no participant flagged

Same rule as the main analysis: control = any record matched to a sentence;
experimental = own record or accepted LLM suggestion; rejected suggestion = not flagged.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rct.data import FILES  # noqa: E402


def n_participants(df):
    matched = df.dropna(subset=["matched_sentenca_id"])
    return matched.groupby("matched_sentenca_id")["participant_code"].nunique()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=ROOT / "data")
    parser.add_argument("--out", default=ROOT / "results" / "error_examples")
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sentences = pd.read_csv(data / FILES["sentences"]).set_index("id")
    reference = pd.read_csv(data / FILES["reference"])
    gold = set(reference["sentenca_id"])
    control = pd.read_csv(data / FILES["control"])
    exp = pd.read_csv(data / FILES["experimental"])

    llm = exp[exp["FL_GERADA_POR_IA"] == 1]
    counts = pd.DataFrame({
        "llm_accepted": n_participants(llm[llm["FL_FALHA_RECUSADA"] == 0]),
        "llm_rejected": n_participants(llm[llm["FL_FALHA_RECUSADA"] == 1]),
        "experimental_own": n_participants(exp[exp["FL_GERADA_POR_IA"] == 0]),
        "control": n_participants(control),
    }).reindex(sentences.index).fillna(0).astype(int)
    counts.insert(0, "page", sentences["pagina"])
    counts.insert(1, "in_reference_standard", counts.index.isin(gold))
    counts["llm_suggested"] = (counts["llm_accepted"] + counts["llm_rejected"]) > 0
    counts["found_by_experimental"] = (counts["llm_accepted"] + counts["experimental_own"]) > 0
    counts["found_by_control"] = counts["control"] > 0
    counts["text"] = sentences["texto_puro"]
    counts.index.name = "sentence_id"
    counts.to_csv(out / "sentence_level_counts.csv")

    false_pos = counts[~counts["in_reference_standard"] & counts["llm_suggested"]]
    false_pos.sort_values("llm_accepted", ascending=False).to_csv(out / "false_positives_llm.csv")

    error_info = reference.groupby("sentenca_id").agg(
        tipo_falha=("tipo_falha", "first"), subtipo=("subtipo", "first"), span_de_erro=("span_de_erro", "first"))
    missed = counts[counts["in_reference_standard"] & ~counts["found_by_experimental"] & ~counts["found_by_control"]]
    missed.join(error_info).to_csv(out / "false_negatives.csv")

    in_ref = counts[counts["in_reference_standard"]]
    print(f"Sentences with reference errors: {len(in_ref)}")
    print(f"  flagged by at least one experimental participant: {int(in_ref['found_by_experimental'].sum())}")
    print(f"  flagged by at least one control participant:      {int(in_ref['found_by_control'].sum())}")
    print(f"  flagged by nobody:                                {len(missed)}")
    print(f"  never suggested by the LLM:                       {int((~in_ref['llm_suggested']).sum())}")
    print(f"LLM suggestions on error-free sentences: {len(false_pos)}")


if __name__ == "__main__":
    main()
