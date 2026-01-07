# Contributing to PyOctaveBand

Thank you for your interest in contributing to PyOctaveBand! We welcome contributions from the community to help improve this project.

## 🛠️ Development Setup

To set up your development environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jmrplens/PyOctaveBand.git
   cd PyOctaveBand
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   Install both production and development dependencies to run tests and linters.
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## ✅ Code Quality Standards

We enforce strict code quality standards. Before submitting a Pull Request, please ensure your code passes the following checks:

### 1. Type Checking (MyPy)
We use strict type checking. Ensure no errors are reported:
```bash
mypy .
```

### 2. Linting & Formatting (Ruff)
We use `ruff` for fast linting and formatting.
```bash
ruff check .
```

### 3. Testing (Pytest)
Run the full test suite to ensure no regressions. We aim for 100% code coverage.
```bash
pytest tests/
```
To check coverage locally:
```bash
pytest --cov=src/pyoctaveband --cov-report=term-missing tests/
```

### 4. Graph & Benchmark Generation (Required)
If you modify the core DSP logic, filter design, or visualization code, you **MUST** regenerate the documentation graphs and the technical benchmark report before submitting your Pull Request. The CI does not perform automatic updates to maintain a clean history.

```bash
# Regenerate documentation graphs
python generate_graphs.py

# Regenerate technical benchmark report
python scripts/benchmark_filters.py
```
Check that the images in `.github/images/` and the `filter_benchmark_report.md` file reflect your changes correctly.

## 🚀 How to Contribute

### Reporting Bugs
If you find a bug, check the [Issues](https://github.com/jmrplens/PyOctaveBand/issues). If not reported, open a new issue with:
- Steps to reproduce
- Expected vs Actual behavior
- Environment details (OS, Python version)

### Pull Requests
1. **Fork** the repository.
2. **Create a branch** for your feature (`git checkout -b feature/amazing-feature`).
3. **Commit** your changes with clear messages.
4. **Verify** your code using the commands above (`pytest`, `mypy`, `ruff`).
5. **Push** to your fork and **Open a Pull Request**.

## License
By contributing to this project, you agree that your contributions will be licensed under the project's [LICENSE](LICENSE) file.
