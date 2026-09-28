# probe-blo

JAX kernel of PROBE (arXiv:2609.30501) on the Figure 1 scalar bilevel toy.

## Install

```bash
pip install -e ".[tests]"
```

## Run

```bash
python -m pytest
python train.py
```

## What does not match the paper

No LLM data-curation experiment and no meta-learning experiment. PGD ball sampling and CG are the paper's inner loops on the Figure 1 scalar toy. Schedules are simplified constants / power-law μ, not the full Appendix B theorem settings.
