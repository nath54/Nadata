"""Release script: sync GitHub, TestPyPI, and PyPI in one run.

Verifies the repository is release-ready (clean git tree, consistent
versions, green tests, strict type check), builds both distributions,
publishes to TestPyPI and then to PyPI, and pushes the release commit
and tag to GitHub.

Credentials are NEVER hardcoded and NEVER passed on the command line
(argv is visible in process listings and shell history). Tokens are
read from environment variables:

    export PYPI_TOKEN=pypi-...        # create at pypi.org
    export TESTPYPI_TOKEN=pypi-...    # create at test.pypi.org

or supplied at invocation with --pypi-token / --testpypi-token (less
safe: shell history) or --pypi-token-file / --testpypi-token-file
(any path you like; the script has no default location).

Usage:

    python scripts/release.py                       # full release
    python scripts/release.py --version 0.1.1       # bump, commit, tag
    python scripts/release.py --skip-testpypi       # skip the dry run
    python scripts/release.py --skip-push           # no git push
    python scripts/release.py --skip-tests          # skip pytest / mypy
"""

# Import Modules
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Optional

# Project root (the folder containing pyproject.toml)
ROOT: Path = Path(__file__).resolve().parent.parent

TESTPYPI_URL: str = "https://test.pypi.org/legacy/"
PYPI_URL: str = "https://upload.pypi.org/legacy/"

_TESTPYPI_JSON: str = "https://test.pypi.org/pypi/nadata/json"
_PYPI_JSON: str = "https://pypi.org/pypi/nadata/json"


def _run(
    args: list[str],
    *,
    env: Optional[dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> str:
    """
    Runs one subprocess, echoing it first.

    'python' resolves to the running interpreter, so the script works
    from any environment (venvs included).

    Args:
        args (list[str]): The command line.
        env (Optional[dict[str, str]], optional): Environment
            override. Defaults to None.
        cwd (Optional[Path], optional): Working directory. Defaults
            to None (the project root).

    Returns:
        str: The captured stdout.
    """

    if args and args[0] == "python":
        args = [sys.executable, *args[1:]]

    print(f"  $ {args[0]} {' '.join(args[1:])}")

    result: Any = subprocess.run(
        args,
        cwd=cwd if cwd is not None else ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)

        raise SystemExit(f"Command failed ({result.returncode})")

    return result.stdout


def _resolve_token(
    *,
    env_var: str,
    cli_token: Optional[str],
    token_file: Optional[Path],
    label: str,
) -> str:
    """
    Resolves one API token: CLI argument, then token file, then
    environment variable.

    Args:
        env_var (str): The environment variable name.
        cli_token (Optional[str]): The --...-token value.
        token_file (Optional[Path]): The --...-token-file value.
        label (str): The index name for messages.

    Returns:
        str: The stripped token.

    Raises:
        SystemExit: If no usable token is provided.
    """

    if cli_token is not None:
        return cli_token.strip()

    if token_file is not None:
        if not token_file.is_file():
            raise SystemExit(f"Token file not found: {token_file}")

        return token_file.read_text(encoding="utf-8").strip()

    token: str = os.environ.get(env_var, "").strip()

    if token:
        return token

    raise SystemExit(
        f"No {label} token. Set the {env_var} environment variable "
        f"(recommended), or pass --{env_var.lower().replace('_', '-')} "
        f"or --{env_var.lower().replace('_', '-').replace('token', 'token-file')}."
    )


def _version_from_pyproject() -> str:
    """
    Reads the project version from pyproject.toml.

    Returns:
        str: The declared version.

    Raises:
        SystemExit: If the version line is missing.
    """

    text: str = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    match: Optional[re.Match[str]] = re.search(
        r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE
    )

    if match is None:
        raise SystemExit("No version found in pyproject.toml")

    return match.group(1)


def _version_from_init() -> str:
    """
    Reads the package version from nadata/__init__.py.

    Returns:
        str: The declared version.

    Raises:
        SystemExit: If the version line is missing.
    """

    text: str = (ROOT / "nadata" / "__init__.py").read_text(
        encoding="utf-8"
    )

    match: Optional[re.Match[str]] = re.search(
        r'^__version__\s*=\s*"([^"]+)"', text, flags=re.MULTILINE
    )

    if match is None:
        raise SystemExit("No __version__ found in nadata/__init__.py")

    return match.group(1)


def _bump_version(version: str) -> None:
    """
    Rewrites both version declarations and commits + tags the release.

    Args:
        version (str): The new version.
    """

    pyproject: Path = ROOT / "pyproject.toml"
    init: Path = ROOT / "nadata" / "__init__.py"

    pyproject.write_text(
        re.sub(
            r'^version\s*=\s*"[^"]+"',
            f'version = "{version}"',
            pyproject.read_text(encoding="utf-8"),
            count=1,
            flags=re.MULTILINE,
        ),
        encoding="utf-8",
    )

    init.write_text(
        re.sub(
            r'^__version__\s*=\s*"[^"]+"',
            f'__version__ = "{version}"',
            init.read_text(encoding="utf-8"),
            count=1,
            flags=re.MULTILINE,
        ),
        encoding="utf-8",
    )

    _run(["git", "add", "pyproject.toml", "nadata/__init__.py"])
    _run(["git", "commit", "-m", f"Release {version}"])
    _run(["git", "tag", f"v{version}"])

    print(f"  version {version} committed and tagged")


def _check_prerequisites() -> str:
    """
    Verifies git state, version consistency, and tool availability.

    Returns:
        str: The current version.
    """

    branch: str = _run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    ).strip()

    if branch != "main":
        raise SystemExit(f"Not on main (on '{branch}')")

    status: str = _run(["git", "status", "--porcelain"])

    if status.strip():
        raise SystemExit(
            "Working tree is not clean - commit or stash first:\n"
            + status
        )

    pyproject_version: str = _version_from_pyproject()
    init_version: str = _version_from_init()

    if pyproject_version != init_version:
        raise SystemExit(
            f"Version mismatch: pyproject.toml has {pyproject_version}, "
            f"nadata/__init__.py has {init_version}"
        )

    _run(["python", "-c", "import build, twine"])

    print(f"  git clean, on main, version {pyproject_version} consistent")

    return pyproject_version


def _check_version_free(version: str, json_url: str, label: str) -> None:
    """
    Aborts when the version is already published on an index.

    Args:
        version (str): The version about to upload.
        json_url (str): The index's project JSON endpoint.
        label (str): The index name for messages.

    Raises:
        SystemExit: If the version already exists.
    """

    try:
        with urllib.request.urlopen(json_url, timeout=30) as response:
            releases: Any = json.load(response).get("releases", {})

    except Exception:
        # Index unreachable: let the upload fail with a real message
        return

    if version in releases:
        raise SystemExit(
            f"{label} already has version {version} - bump the version"
        )

    print(f"  {label}: {version} is free")


def _run_tests() -> None:
    """
    Runs the test suite and the strict type check.
    """

    _run(["python", "-m", "pytest", "tests/", "-q"])
    _run(["python", "-m", "mypy", "nadata/"])


def _build() -> None:
    """
    Rebuilds both distributions from a clean dist/ folder.
    """

    dist: Path = ROOT / "dist"

    if dist.is_dir():
        shutil.rmtree(dist)

    _run(["python", "-m", "build"])


def _upload(token: str, repository_url: str, label: str) -> None:
    """
    Uploads dist/* to one index using env-based credentials.

    The token travels through the environment, never argv (process
    listings and shell history are readable by other local users).

    Args:
        token (str): The API token.
        repository_url (str): The index's upload endpoint.
        label (str): The index name for messages.
    """

    env: dict[str, str] = dict(
        os.environ,
        TWINE_USERNAME="__token__",
        TWINE_PASSWORD=token,
    )

    print(f"  uploading to {label}...")

    _run(
        [
            "python", "-m", "twine", "upload",
            "--repository-url", repository_url,
            "--non-interactive", "dist/*",
        ],
        env=env,
    )


def main() -> int:
    """
    Runs the full release pipeline.

    Returns:
        int: The process exit code.
    """

    parser = argparse.ArgumentParser(
        description="Build, verify, publish (TestPyPI + PyPI), and push."
    )

    parser.add_argument(
        "--version",
        default=None,
        help="Bump both version files to this version, commit and tag.",
    )
    parser.add_argument(
        "--pypi-token",
        default=None,
        help="PyPI API token (prefer the PYPI_TOKEN environment variable).",
    )
    parser.add_argument(
        "--testpypi-token",
        default=None,
        help="TestPyPI API token (prefer TESTPYPI_TOKEN).",
    )
    parser.add_argument(
        "--pypi-token-file",
        type=Path,
        default=None,
        help="File containing the PyPI API token (any path you like).",
    )
    parser.add_argument(
        "--testpypi-token-file",
        type=Path,
        default=None,
        help="File containing the TestPyPI API token.",
    )
    parser.add_argument(
        "--skip-testpypi",
        action="store_true",
        help="Do not upload to TestPyPI.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest and mypy.",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Do not push the commit and tag to GitHub.",
    )

    args: Any = parser.parse_args()

    print("[1/6] Prerequisites")
    if args.version is not None:
        _bump_version(version=args.version)

    version: str = _check_prerequisites()

    if not args.skip_tests:
        print("[2/6] Tests and type check")
        _run_tests()

    print("[3/6] Build")
    _build()
    _run(["python", "-m", "twine", "check", "dist/*"])

    print("[4/6] Version availability")
    _check_version_free(
        version=version, json_url=_PYPI_JSON, label="PyPI"
    )

    if not args.skip_testpypi:
        _check_version_free(
            version=version,
            json_url=_TESTPYPI_JSON,
            label="TestPyPI",
        )

    if not args.skip_testpypi:
        print("[5/6] Publish to TestPyPI")
        _upload(
            token=_resolve_token(
                env_var="TESTPYPI_TOKEN",
                cli_token=args.testpypi_token,
                token_file=args.testpypi_token_file,
                label="TestPyPI",
            ),
            repository_url=TESTPYPI_URL,
            label="TestPyPI",
        )
        print(
            f"  https://test.pypi.org/project/nadata/{version}/"
        )

    print("[6/6] Publish to PyPI")
    _upload(
        token=_resolve_token(
            env_var="PYPI_TOKEN",
            cli_token=args.pypi_token,
            token_file=args.pypi_token_file,
            label="PyPI",
        ),
        repository_url=PYPI_URL,
        label="PyPI",
    )
    print(f"  https://pypi.org/project/nadata/{version}/")

    if not args.skip_push:
        print("[push] GitHub")
        _run(["git", "push", "origin", "main", "--tags"])

    print(f"\nReleased nadata {version}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
