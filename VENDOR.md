# rPPG-Toolbox, vendored

`vendor_rppg/` is not in this repository and has to be cloned before the
backend will start:

```
git clone --depth 1 https://github.com/ubicomplab/rPPG-Toolbox.git vendor_rppg
```

That pulls about 660 MB, nearly all of it pretrained model weights this project
never loads -- only the three modules listed below are imported. It is left out
of version control rather than committed because its own `.git` would make it a
nested repository, which pushes as an empty pointer instead of the files.

Cloned from https://github.com/ubicomplab/rPPG-Toolbox (Liu et al., rPPG-Toolbox,
NeurIPS 2023 Datasets and Benchmarks). It is a repository, not a PyPI package,
so `app/ppg.py` puts this directory on `sys.path` and imports from it directly.

Used from it:

- `unsupervised_methods/methods/POS_WANG.py` — POS (Wang et al., 2017)
- `unsupervised_methods/methods/CHROME_DEHAAN.py` — CHROM (de Haan & Jeanne, 2013)
- `evaluation/post_process.py` — `_calculate_fft_hr` and `_calculate_SNR`

Nothing here is modified. Two accommodations live in `app/ppg.py` instead:

- NumPy 2 removed `np.mat`, which POS_WANG still calls, so the alias is restored
  before the import.
- Both methods take video and average each frame themselves. The app already
  sends colour means, so they are passed as 1x1 frames — the mean of a
  single-pixel frame is that pixel, which reaches the published algorithm with
  no reimplementation.

`pyVHR` was the first choice and is not usable here: it imports `cusignal`, a
RAPIDS package that was discontinued and only ever shipped on conda, leaving a
placeholder 0.0.1 on PyPI.

To update: delete this directory and re-clone, then run
`venv/Scripts/python.exe -m pytest tests/` and `tools/calibrate_snr.py`.
