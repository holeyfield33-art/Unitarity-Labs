# Rank-Syndrome ECC Evidence

`synthetic_sweep.json` is the committed, reproducible evidence artifact for the
classical controlled matrix experiment. Generate it from the repository root:

```bash
python -m pip install -e '.[ecc-research]'
python experiments/rank_syndrome_synthetic.py
```

The recorded generation source SHA is
`486a67c41157ecafb6b2aa590931ead91c6c98b3`. The JSON records that exact SHA,
Python, NumPy, and CVXPY versions. It uses $12\times12$ matrices, primary ranks 1--3, measurements
20/40/60/80/100/120, and seeds 0--19. CLARABEL is the primary CVXPY solver with
absolute gap and feasibility tolerances of `1e-8`; SCS reruns only failed exact
trials. The exact success predicate is relative error recovery `< 1e-5`,
relative state recovery `< 1e-5`, and syndrome residual `< 1e-7`.

The theory reference $m\gtrsim2t(d+T-2t)$ is a generic algebraic
dimension-count criterion, not a nuclear-norm recovery guarantee. Exact
equality solves can yield threshold-edge or solver-dependent outcomes; the
artifact preserves diagnostics and retry classifications instead of treating
all failed outcomes as information-theoretic failures. Noisy parity trials use
known injected syndrome-noise bounds and a separate fixed reporting threshold.