# Set shell for Windows compatibility
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
set positional-arguments
# Run tests (usage: just t or just t anthropic)
# @t *ARGS='':
#     #!/usr/bin/env python3
#     import sys
#     from pathlib import Path
#     import subprocess

#     args = "{{ARGS}}"
#     cmd = f"tests/test_{args}.py" if args else "tests/"

#     venv = Path('.venv')
#     pytest = venv / ('Scripts' if sys.platform == 'win32' else 'bin') / 'pytest'
#     subprocess.run([str(pytest), cmd, "--tb=no", "-rA"], check=True)

t:
    pytest -xvs

# Default recipe that sets up everything
setup: _ensure-venv _install-deps
    @echo "Development environment ready!"

# Create virtualenv if it doesn't exist
_ensure-venv:
    #!/usr/bin/env python3
    from pathlib import Path
    import venv
    venv_dir = Path('.venv')
    if not venv_dir.exists():
        print("Creating virtual environment...")
        venv.create(venv_dir, with_pip=True)

# Install dependencies in dev mode
_install-deps:
    #!/usr/bin/env python3
    from pathlib import Path
    import subprocess
    import sys

    venv_dir = Path('.venv')
    pip_path = venv_dir / ('Scripts' if sys.platform == 'win32' else 'bin') / 'pip'

    print("Installing package in development mode...")
    subprocess.run([str(pip_path), "install", "-e", ".[all]"], check=True)

# Run all linting checks
lint:
    .venv/bin/black .

    .venv/bin/isort  .
    .venv/bin/pyright .

# Format code
format:
    .venv/bin/black prompter tests
    .venv/bin/isort prompter tests

# Clean up generated files
clean:
    rm -rf .venv
    rm -rf *.egg-info
    rm -rf build
    rm -rf dist
    rm -rf __pycache__
    rm -rf .pytest_cache
    rm -rf .mypy_cache
    rm -rf docs/_build

# Build sphinx docs
docs:
    cd docs && make html