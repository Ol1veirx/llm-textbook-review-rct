"""Reproduce the tables and figures of the paper.

Usage (from the repository root):
    python scripts/run_all.py [--data data] [--out results]

Tables are written as CSV and LaTeX to <out>/tables, figures to <out>/figures.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rct import figures, tables  # noqa: E402
from rct.data import load_study  # noqa: E402
from rct.metrics import participant_metrics, to_itt  # noqa: E402


def save(df, folder, name, index=False):
    df.to_csv(folder / f"{name}.csv", index=index)
    df.to_latex(folder / f"{name}.tex", index=index, escape=True, float_format="%.4f")
    print(f"  {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=ROOT / "data")
    parser.add_argument("--out", default=ROOT / "results")
    args = parser.parse_args()

    out = Path(args.out)
    tab_dir, fig_dir = out / "tables", out / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    study = load_study(args.data)
    per_protocol = participant_metrics(study)               # participants with at least one record
    itt = to_itt(per_protocol, study.n_randomized)          # all randomized participants (40 / 38)
    itt.to_csv(out / "participant_metrics_itt.csv", index=False)

    print("Tables")
    save(tables.table2_reference_standard(study), tab_dir, "table2_reference_standard")

    descriptives, tests = tables.table3_descriptives(itt)
    save(descriptives, tab_dir, "table3_descriptives", index=True)
    save(tests, tab_dir, "table3_tests")

    save(tables.table4_quality(itt), tab_dir, "table4_accuracy_recall_f1_kappa")

    main_rows, top_rows, exposure_rows, mcnemar_rows = tables.table5_ai_vs_assisted(study)
    save(main_rows, tab_dir, "table5_ai_only_vs_assisted")
    save(top_rows, tab_dir, "table5_top5_educators")
    save(exposure_rows, tab_dir, "table5_per_educator_exposure")
    save(mcnemar_rows, tab_dir, "table5_mcnemar")

    save(tables.table7_sensitivity(per_protocol, itt), tab_dir, "table7_sensitivity")
    save(tables.participation(study, per_protocol), tab_dir, "participation")
    save(tables.spearman(itt), tab_dir, "spearman")

    print("Figures")
    figures.fig5_boxplots(itt, fig_dir / "fig5_boxplots.png")
    figures.fig6_scatter(itt, fig_dir / "fig6_scatter.png")
    print(f"Done. Results in {out}")


if __name__ == "__main__":
    main()
