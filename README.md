# LLM-assisted linguistic review of textbooks: analysis code

Code for the paper *Large Language Model-Assisted Linguistic Review in a Governmental
Textbook Approval Process: A Randomized Controlled Study*.

In the study, 78 educators looked for orthographic and grammatical errors in 20 pages
of a Brazilian Portuguese textbook (585 sentences). 40 worked on their own (control) and
38 worked with suggestions from an LLM (experimental). This repository turns the response
logs into the tables and figures of the paper.

- Data: https://doi.org/10.5281/zenodo.23395396

## Quick start

Tested with Python 3.13.

```bash
pip install -r requirements.txt
python scripts/download_data.py   # downloads the data from Zenodo into data/ and checks it
python scripts/run_all.py         # tables and figures, about 10 seconds
python scripts/error_examples.py  # false positives and false negatives
pytest                            # checks the output against the paper
```

Run these from the folder of this repository. Results are written to `results/`.
`make all` runs the last three steps in one go. If you prefer to download the data by hand,
see [data/README.md](data/README.md). The error examples contain textbook sentences, so they
are not committed (`results/error_examples/` is in `.gitignore`).

## What each output is

Numbers refer to the tables and figures of the submitted manuscript.

| Paper | File in `results/` |
|---|---|
| Table 2, reference standard | `tables/table2_reference_standard.csv` |
| Table 3, responses and precision | `tables/table3_descriptives.csv`, `table3_tests.csv` |
| Table 4, accuracy, recall, F1, kappa | `tables/table4_accuracy_recall_f1_kappa.csv` |
| Table 5, AI-only vs AI-assisted | `tables/table5_ai_only_vs_assisted.csv` |
| Table 5, top-5 educators | `tables/table5_top5_educators.csv`, `table5_mcnemar.csv` |
| Extra: sum over each educator's own pages | `tables/table5_per_educator_exposure.csv` |
| Table 7, sensitivity | `tables/table7_sensitivity.csv` |
| Participation (14/40 vs 1/38) | `tables/participation.csv` |
| Spearman correlations | `tables/spearman.csv` |
| Figure 5, boxplots | `figures/fig5_boxplots.png` |
| Figure 6, scatter plots | `figures/fig6_scatter.png` |

Every table is saved as `.csv` and `.tex`. `participant_metrics_itt.csv` has the metrics of
each participant. For a guided read of the same steps, open
[notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb).

## How the metrics are defined

- **Unit.** The sentence. A sentence is erroneous if the reference standard lists at least
  one error in it (100 of 585).
- **Flagged sentence.** Control: any record matched to a sentence. Experimental: the
  educator's own records plus the LLM suggestions they accepted. A rejected suggestion
  counts as not flagged.
- **Exposure.** The task was time limited, so people stopped at different pages. Recall,
  F1, accuracy and kappa are computed only on sentences up to the last page where the
  participant registered a record. Precision does not change with exposure.
- **No records.** 15 participants (14 control, 1 experimental) registered nothing, so
  their exposure is unknown. The main analysis keeps all randomized participants
  (40 and 38) and gives these 15 a score of 0 in every metric. Table 7 also shows the
  results without them.
- **Tests.** Mann-Whitney U (two-sided). p-values for the four quality metrics are adjusted
  with Holm. Outliers (Table 7) are found with Tukey's fences on total responses, within
  each group.
- **AI-only vs AI-assisted.** AI-only is every sentence the LLM flagged. AI-assisted is the
  union of the sentences flagged by the experimental educators, so it describes the group,
  not one person. Adding a participant who flagged nothing does not change a union.
  Individual educators in the top-5 table are ranked by their own F1-score. The comparison
  with the LLM uses an exact McNemar test.

## What is not here

The ANCOVA on the pre-test and post-test (Table 6), the questionnaire and the survey
figure use data that we do not publish for privacy reasons. They cannot be reproduced from
this repository.

## Layout

```
data/        the CSV files go here (not stored in git)
src/rct/     data.py (loading, flagged sentences, exposure), metrics.py, tables.py, figures.py
scripts/     download_data.py, run_all.py, error_examples.py
tests/       compares the output with the numbers in the paper
notebooks/   walkthrough.ipynb
results/     written by the scripts
```

## Notes

- Participants are identified by random codes (`P001` to `P078`).
- Versions of the main packages are pinned in `requirements.txt`. Newer versions should
  work, but the tests were run with these.

## License and citation

The code is released under the MIT license (see `LICENSE`). If you use it, please cite the
paper and this repository (see `CITATION.cff`), and the dataset above.
