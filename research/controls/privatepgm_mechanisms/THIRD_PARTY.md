# Third-party files

`aim.py`, `mst.py` and `mechanism.py` come from Ryan McKenna's private-pgm (https://github.com/ryan112358/private-pgm,
Apache-2.0). `cdp2adp.py` comes from IBM's discrete-gaussian-differential-privacy (Apache-2.0; its licence header is
kept). They are included so that the reference AIM and MST baselines in the paper can be rerun.

Changes made here: `aim.py`, `mst.py` and `mechanism.py` each add a `sys.path` line so that they import their
neighbours from this folder, and `aim.py` imports `mbi` before its module docstring. Nothing else was changed.
The wrapper that runs them, `../pgm_baselines.py`, is part of this repository.
