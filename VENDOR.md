# rPPG-Toolbox, vendored

`vendor_rppg/` is committed to this repository, so a plain `git clone` is enough.
Only the files the app imports are kept (listed below, plus the upstream
LICENSE), about 50 KB, instead of the full 660 MB upstream repository, which is
mostly pretrained weights this project never loads.

Upstream commit: b7500b848f84ad7f86e277b4612563b69f4f88f9.

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

One change to upstream: `unsupervised_methods/__init__.py` is emptied, because
the original star-imports methods that are not vendored here.

To update: sparse-clone upstream (`--filter=blob:none --sparse`, then
`git sparse-checkout set unsupervised_methods evaluation`), copy the files listed
above over this directory, re-empty that `__init__.py`, then run
`venv/Scripts/python.exe -m pytest tests/` and `tools/calibrate_snr.py`.
