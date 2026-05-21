# Contributing to XAI-SDN

Thank you for your interest in contributing! This document covers everything
you need to know to get started.

---

## Development Setup

```bash
git clone https://github.com/adeliusa486/XAI-SDN.git
cd xai-sdn
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
mkdir -p data model/artifacts logs
```

## Branching

- `main` — stable, tested releases only
- `develop` — active development; open PRs against this branch
- Feature branches: `feature/<short-description>`
- Bug fix branches: `fix/<issue-number>-short-description`

## Pull Request Process

1. Fork the repository and create your branch from `develop`.
2. Write tests for any new functionality.
3. Ensure `make test` passes.
4. Ensure `make lint` and `make format-check` pass.
5. Update documentation if you're changing behaviour.
6. Open a PR with a clear title and description.

## Code Style

- **Formatter**: `black` (line length 100)
- **Import sorter**: `isort` (black profile)
- **Linter**: `flake8`
- **Type hints**: required for all public functions and class methods
- **Docstrings**: Google style for public APIs

```bash
make format        # auto-format
make lint          # check style
make type-check    # mypy
```

## Testing

```bash
make test          # full suite
make test-cov      # with coverage report
make smoke-test    # quick smoke test
```

All PRs must have ≥ 60% coverage on changed modules.

## Reporting Issues

Use GitHub Issues. Include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behaviour
- Relevant logs (redact any sensitive network data)

## Feature Requests

Open a GitHub Discussion before implementing large features — alignment
with the research scope of the project is important.

## Areas Needing Contribution

- [ ] WebSocket streaming for real-time alert feed
- [ ] Adversarial robustness evaluation (IP rotation attacks)
- [ ] Additional datasets (CIC-IDS2018, CAIDA)
- [ ] Kubernetes Helm chart improvements
- [ ] React frontend alternative to Streamlit
- [ ] Adaptive entropy window sizing
- [ ] OpenFlow TLS setup guide
- [ ] CICFlowMeter full integration via network tap
