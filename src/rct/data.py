"""Load the released CSV files and build the objects the analysis needs.

The main ideas (all documented in the README):

* A participant's *positive responses* are the sentences they flagged as erroneous.
  Control: any record that could be matched to a sentence.
  Experimental: their own records, plus LLM suggestions they accepted.
  A rejected LLM suggestion counts as "not flagged".
* A participant's *exposure* is every sentence up to the last page where they
  registered a record. Recall, F1, accuracy and kappa are computed only inside
  that window.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("Control", "Experimental")

FILES = {
    "sentences": "sentences.csv",
    "reference": "reference_standard.csv",
    "participants": "participants.csv",
    "control": "responses_control.csv",
    "experimental": "responses_experimental.csv",
}


@dataclass
class Study:
    sentences: pd.DataFrame       # one row per sentence: id, pagina, texto_puro
    reference: pd.DataFrame       # one row per reference error
    participants: pd.DataFrame    # participant_code, arm, has_records
    responses: dict               # arm -> raw response log
    all_ids: list                 # sentence ids, sorted
    page_of: dict                 # sentence id -> page
    gold: set                     # sentences with at least one reference error
    positives: dict               # (arm, participant) -> set of flagged sentence ids
    llm_flagged: set              # sentences the LLM flagged, whatever the educator decided
    exposure: pd.DataFrame        # arm, participant, last_page, n_records
    n_randomized: dict            # arm -> number of randomized participants

    def universe_upto(self, page):
        """Sentences on pages <= page."""
        return {i for i in self.all_ids if self.page_of[i] <= page}


def _positive_responses(df, arm):
    if arm == "Experimental":
        by_llm = df["FL_GERADA_POR_IA"].fillna(0).astype(int) == 1
        rejected = df["FL_FALHA_RECUSADA"].fillna(0).astype(int) == 1
        df = df[~(by_llm & rejected)]
    df = df[df["matched_sentenca_id"].notna()]
    return {p: set(g["matched_sentenca_id"].astype(int)) for p, g in df.groupby("participant_code")}


def _exposure(df, arm, page_of):
    d = df.copy()
    sentence_page = pd.to_numeric(d["matched_sentenca_id"], errors="coerce").map(
        lambda v: page_of.get(int(v)) if pd.notna(v) else np.nan)
    d["page_seen"] = pd.concat([d["CO_PAGINA_FALHA"], sentence_page], axis=1).max(axis=1)
    out = (d.groupby("participant_code")
             .agg(last_page=("page_seen", "max"), n_records=("page_seen", "size"))
             .reset_index().rename(columns={"participant_code": "participant"}))
    out["last_page"] = out["last_page"].astype(int)
    out.insert(0, "arm", arm)
    return out


def load_study(data_dir):
    data_dir = Path(data_dir)
    missing = [f for f in FILES.values() if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing in {data_dir}: {', '.join(missing)}. See data/README.md to download them.")

    sentences = pd.read_csv(data_dir / FILES["sentences"])
    reference = pd.read_csv(data_dir / FILES["reference"])
    participants = pd.read_csv(data_dir / FILES["participants"])
    responses = {"Control": pd.read_csv(data_dir / FILES["control"]),
                 "Experimental": pd.read_csv(data_dir / FILES["experimental"])}

    all_ids = sorted(sentences["id"].astype(int))
    page_of = dict(zip(sentences["id"].astype(int), sentences["pagina"].astype(int)))
    gold = set(reference["sentenca_id"].dropna().astype(int))
    assert gold <= set(all_ids), "Reference standard has sentence ids that are not in sentences.csv"

    positives = {}
    for arm in ARMS:
        for participant, ids in _positive_responses(responses[arm], arm).items():
            positives[(arm, participant)] = ids

    exp = responses["Experimental"]
    llm = exp[exp["FL_GERADA_POR_IA"].fillna(0).astype(int) == 1]
    llm_flagged = set(pd.to_numeric(llm["matched_sentenca_id"], errors="coerce").dropna().astype(int))

    exposure = pd.concat([_exposure(responses[a], a, page_of) for a in ARMS], ignore_index=True)
    n_randomized = {"Control": int((participants["arm"] == "control").sum()),
                    "Experimental": int((participants["arm"] == "experimental").sum())}

    return Study(sentences, reference, participants, responses, all_ids, page_of, gold,
                 positives, llm_flagged, exposure, n_randomized)
