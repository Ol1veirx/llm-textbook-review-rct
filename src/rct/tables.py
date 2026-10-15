"""One function per table of the paper. Each returns a DataFrame."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from .data import ARMS
from .metrics import confusion_counts, drop_outliers, metrics_from_counts

QUALITY = ["Precision", "Recall", "F1", "Kappa"]


# ---------------------------------------------------------------- helpers
def fmt_p(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def _groups(df, metric):
    """Values of a metric for the control and the experimental group."""
    return (df.loc[df["Arm"] == "Control", metric].to_numpy(float),
            df.loc[df["Arm"] == "Experimental", metric].to_numpy(float))


def _mann_whitney(df, metric):
    x, y = _groups(df, metric)
    result = stats.mannwhitneyu(x, y, alternative="two-sided")
    return result.statistic, result.pvalue


def _se(a):
    return a.std(ddof=1) / np.sqrt(len(a))


def _holm(df, metrics):
    p = [_mann_whitney(df, m)[1] for m in metrics]
    return dict(zip(metrics, multipletests(p, method="holm")[1]))


# ---------------------------------------------------------------- Table 2
SUBTYPE_LABELS = {
    "Ortografia": {
        "Pontuação": "Punctuation", "Maiúscula/Minúscula": "Capitalization", "Digitação": "Typing",
        "Repetição Indevida": "Undue repetition", "Hífen": "Hyphen"},
    "Gramática": {
        "Pronome": "Pronoun use", "Regência": "Regency", "Concordância Nominal": "Nominal agreement",
        "Concordância Verbal": "Verbal agreement", "Paralelismo": "Parallelism",
        "Artigo": "Article use", "artigo": "Article use", "ausência de artigo": "Article use",
        "falta de artigo": "Article use", "Coesão": "Cohesion"},
}


def table2_reference_standard(study):
    """Composition of the reference standard: sentences, error types and subtypes."""
    ref = study.reference
    n_sent, n_err_sent = len(study.all_ids), len(study.gold)
    rows = [("Excerpt", None, None),
            ("Pages", study.sentences["pagina"].nunique(), None),
            ("Sentences", n_sent, 100.0),
            ("Sentences with at least one error", n_err_sent, 100 * n_err_sent / n_sent),
            ("Error-free sentences", n_sent - n_err_sent, 100 * (n_sent - n_err_sent) / n_sent),
            (f"Errors (total = {len(ref)})", None, None)]

    for tipo, label in [("Ortografia", "Orthographic"), ("Gramática", "Grammatical")]:
        n = int((ref["tipo_falha"] == tipo).sum())
        rows.append((label, n, 100 * n / len(ref)))

    for tipo, label in [("Ortografia", "Orthographic"), ("Gramática", "Grammatical")]:
        sub = ref[ref["tipo_falha"] == tipo]["subtipo"].map(SUBTYPE_LABELS[tipo]).fillna("Other")
        counts = sub.value_counts()
        rows.append((f"{label} subtypes", None, None))
        for name in [n for n in counts.index if n != "Other"] + ["Other"]:
            if name in counts:
                rows.append((name, int(counts[name]), 100 * counts[name] / len(sub)))

    per_sentence = ref.groupby("sentenca_id").size()
    rows.append(("Errors per erroneous sentence", None, None))
    for label, n in [("1 error", (per_sentence == 1).sum()), ("2 errors", (per_sentence == 2).sum()),
                     ("3 or more errors", (per_sentence >= 3).sum())]:
        rows.append((label, int(n), 100 * n / n_err_sent))

    out = pd.DataFrame(rows, columns=["Characteristic", "n", "%"])
    out["n"] = out["n"].astype("Int64")
    return out.round(1)


# ---------------------------------------------------------------- Table 3
def table3_descriptives(itt):
    """Total responses and precision: descriptive statistics and Mann-Whitney U tests."""
    names = ["Mean", "SD", "Minimum", "1st Quartile (25%)", "Median (50%)", "3rd Quartile (75%)", "Maximum"]

    def describe(s):
        return [s.mean(), s.std(ddof=1), s.min(), s.quantile(.25), s.median(), s.quantile(.75), s.max()]

    table = pd.DataFrame(index=names)
    for arm in ARMS:
        part = itt[itt["Arm"] == arm]
        table[f"{arm} - Total Responses"] = [f"{v:.2f}" for v in describe(part["Responses"])]
        table[f"{arm} - Precision"] = [f"{100 * v:.2f}%" for v in describe(part["Precision"])]

    tests = pd.DataFrame(
        [{"Variable": v, "U": _mann_whitney(itt, v)[0], "p": fmt_p(_mann_whitney(itt, v)[1])}
         for v in ["Responses", "Precision"]])
    return table, tests


# ---------------------------------------------------------------- Table 4
def table4_quality(itt, metrics=("Accuracy", "Recall", "F1", "Kappa")):
    """Mean (SE) by group, Mann-Whitney U and Holm-adjusted p over the listed metrics."""
    metrics = list(metrics)
    holm = _holm(itt, metrics)
    rows = []
    for m in metrics:
        x, y = _groups(itt, m)
        if m == "Kappa":
            fmt = lambda a: f"{a.mean():.4f} ({_se(a):.4f})"
        else:
            fmt = lambda a: f"{100 * a.mean():.2f}% ({100 * _se(a):.2f})"
        label = "F1-Score" if m == "F1" else m
        rows.append({"Metric": f"Mean {label} (SE)", "Control": fmt(x), "Experimental": fmt(y),
                     "U": round(_mann_whitney(itt, m)[0], 1), "p-value (Holm)": fmt_p(holm[m])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Table 5
def _row(name, flagged, study, universe):
    m = metrics_from_counts(*confusion_counts(flagged, study.gold, universe))
    return {"Condition": name, "Accuracy": m["Accuracy"], "Precision": m["Precision"],
            "Recall": m["Recall"], "F1-Score": m["F1"], "Cohen's Kappa": m["Kappa"]}


def _mcnemar_vs_llm(flagged, study, universe):
    """Exact McNemar test (sentence level) of a set of flagged sentences against the LLM alone."""
    ids = sorted(universe)
    llm = np.array([i in study.llm_flagged for i in ids])
    other = np.array([i in flagged for i in ids])
    only_llm, only_other = int((llm & ~other).sum()), int((~llm & other).sum())
    table = [[int((~llm & ~other).sum()), only_other], [only_llm, int((llm & other).sum())]]
    return only_llm, only_other, mcnemar(table, exact=True).pvalue


def _experimental_with_records(study):
    return [p for a, p in study.exposure[["arm", "participant"]].itertuples(index=False) if a == "Experimental"]


def table5_ai_vs_assisted(study, top_n=5):
    """AI-only versus the pooled experimental educators, and the top-n individual educators.

    * AI-Assisted (pooled) is the union of the sentences flagged by all experimental
      educators. A participant with no flagged sentence adds nothing to the union, so
      the pooled row is the same with 37 educators (with records) or 38 (all randomized).
    * Educators are ranked by their own F1-score over the 585 sentences
      (ties: recall, then precision, then participant code).
    * 'Per-educator exposure' sums each educator's confusion matrix inside the pages
      that educator reached.
    """
    universe = set(study.all_ids)
    educators = _experimental_with_records(study)
    flagged = {p: study.positives.get(("Experimental", p), set()) for p in educators}
    pooled = set().union(*flagged.values())
    n_with, n_all = len(educators), study.n_randomized["Experimental"]
    assert max(study.page_of[i] for i in pooled | study.llm_flagged) == max(study.page_of.values())

    ranking = pd.DataFrame([{"participant": p, **metrics_from_counts(*confusion_counts(s, study.gold, universe))}
                            for p, s in flagged.items()])
    ranking = ranking.sort_values(["F1", "Recall", "Precision", "participant"],
                                  ascending=[False, False, False, True]).reset_index(drop=True)

    main = [_row("AI-Only", study.llm_flagged, study, universe),
            _row(f"AI-Assisted (pooled, n={n_with})", pooled, study, universe),
            _row(f"AI-Assisted (pooled, n={n_all})", pooled, study, universe)]

    top = [_row("AI-Only", study.llm_flagged, study, universe),
           _row(f"AI-Assisted (pooled, n={n_with})", pooled, study, universe)]
    tests = [("AI-Assisted (pooled)", *_mcnemar_vs_llm(pooled, study, universe))]
    for i in range(top_n):
        p = ranking.loc[i, "participant"]
        top.append(_row(f"Educator {i + 1}", flagged[p], study, universe))
        tests.append((f"Educator {i + 1}", *_mcnemar_vs_llm(flagged[p], study, universe)))

    # Sum of each educator's confusion matrix inside their own exposure window.
    last_page = study.exposure.set_index(["arm", "participant"])["last_page"]
    total = np.zeros(4, dtype=int)
    for p in educators:
        window = study.universe_upto(int(last_page[("Experimental", p)]))
        total += np.array(confusion_counts(flagged[p], study.gold, window))
    m = metrics_from_counts(*total)
    per_exposure = [{"Condition": f"AI-Assisted (per-educator exposure, n={n_with})", "Accuracy": m["Accuracy"],
                     "Precision": m["Precision"], "Recall": m["Recall"], "F1-Score": m["F1"],
                     "Cohen's Kappa": m["Kappa"]}]

    mcnemar_table = pd.DataFrame(tests, columns=["Comparison with AI-Only", "LLM only", "Educators only", "McNemar p"])
    mcnemar_table["McNemar p"] = mcnemar_table["McNemar p"].map(fmt_p)
    return (pd.DataFrame(main).round(4), pd.DataFrame(top).round(4),
            pd.DataFrame(per_exposure).round(4), mcnemar_table)


# ---------------------------------------------------------------- Table 7
def table7_sensitivity(per_protocol, itt):
    """Group means under four samples: all randomized or only participants with records, with and without outliers."""
    itt_clean, _ = drop_outliers(itt)
    pp_clean, _ = drop_outliers(per_protocol)
    samples = [("All randomized", itt), ("All, no outliers", itt_clean),
               (">=1 record", per_protocol), (">=1 record, no outliers", pp_clean)]
    rows = []
    for name, df in samples:
        n = df.groupby("Arm").size().reindex(list(ARMS)).tolist()
        holm = _holm(df, QUALITY)
        row = {"Sample": f"{name} ({n[0]}/{n[1]})"}
        for m in QUALITY:
            x, y = _groups(df, m)
            row[m] = f"{x.mean():.3f} / {y.mean():.3f}" if m == "Kappa" else f"{100 * x.mean():.2f} / {100 * y.mean():.2f}"
        row["p (Holm, range)"] = f"{fmt_p(min(holm.values()))} - {fmt_p(max(holm.values()))}"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- participation
def participation(study, per_protocol):
    """Participants without any record, pages reached and Fisher's exact test."""
    part = study.participants
    no_records = {a: int(((part["arm"] == a.lower()) & (part["has_records"] == 0)).sum()) for a in ARMS}
    n = study.n_randomized
    odds, p = stats.fisher_exact([[no_records["Control"], n["Control"] - no_records["Control"]],
                                  [no_records["Experimental"], n["Experimental"] - no_records["Experimental"]]])
    pages = (study.exposure.assign(pages=lambda d: d["last_page"] - min(study.page_of.values()) + 1)
             .groupby("arm")["pages"].median())
    out = pd.DataFrame({"Arm": list(ARMS),
                        "Randomized": [n[a] for a in ARMS],
                        "No records": [no_records[a] for a in ARMS],
                        "% no records": [round(100 * no_records[a] / n[a], 1) for a in ARMS],
                        "Median pages reached (with records)": [pages[a] for a in ARMS]})
    out["Fisher exact p"] = [fmt_p(p), ""]
    return out


def spearman(itt):
    """Spearman correlation between total responses and correct responses, by group."""
    rows = []
    for arm in ARMS:
        g = itt[itt["Arm"] == arm]
        rho, p = stats.spearmanr(g["Responses"], g["Correct"])
        rows.append({"Arm": arm, "Spearman rho": round(rho, 4), "p": p})
    return pd.DataFrame(rows)
