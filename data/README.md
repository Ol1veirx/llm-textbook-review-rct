# Data

The data are not stored in this repository. The easiest way is `python scripts/download_data.py`,
which downloads and checks the files. To do it by hand, download them from Zenodo
(https://doi.org/10.5281/zenodo.23395396) and put these five files in this folder:

```
data/sentences.csv                 585 sentences with their page
data/reference_standard.csv        the 142 reference errors (one row per error)
data/participants.csv              participant code, group, whether they registered any record
data/responses_control.csv         response log, control group
data/responses_experimental.csv    response log, experimental group
```

Check that the files are intact with `sha256sum -c SHA256SUMS.txt` (run it inside this folder), using the checksum file
from the Zenodo record. Column descriptions are in `DATA_DICTIONARY.md`, also in the record.

The textbook excerpts in `sentences.csv` are included only so the results can be verified.
They belong to their rights holders.
