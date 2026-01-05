# Contributing to PyOctaveBand

Thank you for your interest in contributing to PyOctaveBand! We welcome contributions from the community to help improve this project.

## How to Contribute

### Reporting Bugs
If you find a bug, please check the [Issues](https://github.com/jmrplens/PyOctaveBand/issues) to see if it has already been reported. If not, open a new issue with a clear description of the problem, including:
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, etc.)

### Suggesting Enhancements
If you have an idea for a new feature or improvement, please open an issue to discuss it before starting work. This helps ensure that your contribution aligns with the project's goals.

### Pull Requests
1. **Fork the repository** to your own GitHub account.
2. **Clone the repository** to your local machine.
3. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b my-feature-branch
   ```
4. **Make your changes** and ensure the code follows the existing style.
5. **Run tests** to ensure your changes don't break existing functionality:
   ```bash
   python tests/test_basic.py
   python tests/test_multichannel.py
   python tests/test_audio_processing.py
   ```
6. **Commit your changes** with descriptive commit messages.
7. **Push to your branch**:
   ```bash
   git push origin my-feature-branch
   ```
8. **Open a Pull Request** against the `master` or `main` branch of the original repository.

## Development Setup

To set up your development environment:

1. Clone the repo.
2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## License
By contributing to this project, you agree that your contributions will be licensed under the project's [LICENSE](LICENSE) file.
