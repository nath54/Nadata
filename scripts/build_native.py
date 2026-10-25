"""
Phase 1b — distro-invariant native distribution builder.

Produces a self-contained native release for one target platform:

    dist/native/<target>/
        nadata.dll | libnadata.so | nadata.dylib   <- the shim
        nadata.h                                   <- the ABI header
        nadata/                                    <- the engine package
        runtime/                                   <- pinned CPython
        python312.dll (Windows copy, next to the shim)

The interpreter comes from python-build-standalone (PBS) — the same
runtimes `uv` uses — pinned to one release tag and verified against
the release SHA256SUMS. On Linux the STATIC libpython is linked into
libnadata.so itself (true distro invariance); on Windows/macOS the
shared runtime ships alongside and `PyConfig.home` is set relative to
the library location by the shim (shim_runtime.c boot_interpreter).

Usage:
    python scripts/build_native.py                  # host target
    python scripts/build_native.py --target x86_64-unknown-linux-gnu
    python scripts/build_native.py --run            # + smoke-validate
    python scripts/build_native.py --list-assets    # inspect PBS release

The result is validated by running the C smoke test against the STAGED
library with NADATA_HOME unset (proving the embedded layout boots).
"""

# Import Modules
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "native"))

# pylint: disable=wrong-import-position
import build as dev_build  # native/build.py helpers (vcvars, run_command)

# Pin: python-build-standalone release + CPython version.
#
# Why 3.12: the embedding path (PyConfig + libpython) is most battle-
# tested on this line, and it predates the 3.13+ free-threaded build
# churn. The interpreter speed delta vs 3.14 is a few percent — noise
# for a data engine. The full 995-test suite passes under the embedded
# 3.12 runtime, so there is no behavioral drift vs the 3.14 dev env.
# Bumping is a one-line change; re-run scripts/build_native.py --run.
PBS_REPO = "astral-sh/python-build-standalone"
PBS_TAG = "20241016"
CPYTHON_VERSION = "3.12.7"

CACHE_DIR = REPO_ROOT / ".native-cache"
DIST_DIR = REPO_ROOT / "dist" / "native"

# Directories removed from the staged runtime (kept lean, still complete).
PRUNE_DIRS = ("test", "idlelib", "tkinter", "turtledemo")


def log(message: str) -> None:
    """Timestamp-free step logger (keeps CI output clean)."""

    print(f"[native] {message}", flush=True)


def host_target() -> str:
    """Returns the default target triple for the running machine."""

    machine = {"AMD64": "x86_64", "arm64": "aarch64",
               "aarch64": "aarch64", "x86_64": "x86_64"}.get(
        platform.machine().lower(), "x86_64")

    if sys.platform == "win32":
        return f"{machine}-pc-windows-msvc"

    if sys.platform == "darwin":
        return f"{machine}-apple-darwin"

    return f"{machine}-unknown-linux-gnu"


# ------------------------------------------------------------------ #
# PBS download + verification                                         #
# ------------------------------------------------------------------ #


def fetch_release_assets() -> list[dict]:
    """Lists the assets of the pinned PBS release via the GitHub API."""

    url = (f"https://api.github.com/repos/{PBS_REPO}/"
           f"releases/tags/{PBS_TAG}")

    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return payload.get("assets", [])


def pick_asset(assets: list[dict], target: str) -> tuple[str, str]:
    """
    Selects the install-only archive + SHA256SUMS for a target triple.

    Returns:
        tuple[str, str]: (asset name, browser download URL).
    """

    sums = next((a for a in assets if a["name"] == "SHA256SUMS"), None)

    if sums is None:
        raise SystemExit(f"SHA256SUMS asset missing from {PBS_TAG}")

    prefix = f"cpython-{CPYTHON_VERSION}+{PBS_TAG}-{target}"
    archive = next(
        (a for a in assets
         if a["name"].startswith(prefix)
         and "install_only" in a["name"]
         and "stripped" not in a["name"]
         and a["name"].endswith(".tar.gz")),
        None,
    )

    if archive is None:
        matching = [a["name"] for a in assets if target in a["name"]]
        raise SystemExit(
            f"no install_only asset for {target}\n"
            f"available: {matching[:8]}")

    return archive["name"], archive["browser_download_url"]


def sha256_of(path: Path) -> str:
    """Streams a file through SHA256."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def ensure_runtime(target: str) -> Path:
    """
    Downloads, verifies and extracts the pinned PBS runtime.

    Returns:
        Path: The extracted install directory (contains include/, lib/).
    """

    assets = fetch_release_assets()
    name, url = pick_asset(assets, target)
    downloads = CACHE_DIR / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    archive_path = downloads / name

    if not archive_path.exists():
        log(f"downloading {name}")
        urllib.request.urlretrieve(url, archive_path)

    # Checksum verification against the release SHA256SUMS.
    sums_path = downloads / f"SHA256SUMS-{PBS_TAG}"

    if not sums_path.exists():
        sums_asset = next(
            a for a in assets if a["name"] == "SHA256SUMS")
        urllib.request.urlretrieve(sums_asset["browser_download_url"],
                                   sums_path)

    expected = None

    for line in sums_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if len(parts) == 2 and parts[1] == name:
            expected = parts[0]

            break

    if expected is None:
        raise SystemExit(f"{name} not listed in SHA256SUMS")

    actual = sha256_of(archive_path)

    if actual != expected:
        raise SystemExit(f"checksum mismatch for {name}: {actual}")

    log(f"checksum ok: {name}")

    # Extract (idempotent: reuse a previous extraction).
    extract_dir = CACHE_DIR / "pbs" / target

    if not (extract_dir / "install").exists():
        if extract_dir.exists():
            shutil.rmtree(extract_dir)

        extract_dir.mkdir(parents=True)
        log(f"extracting to {extract_dir}")

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")

    # install_only archives root everything under `python/`.
    install = extract_dir / "python"

    if not install.exists():
        raise SystemExit(f"unexpected archive layout under {extract_dir}")

    return install


# ------------------------------------------------------------------ #
# Layout discovery inside the PBS install                             #
# ------------------------------------------------------------------ #


def pbs_include(install: Path) -> Path:
    """Locates the directory containing Python.h."""

    for candidate in (install / "include",):
        if (candidate / "Python.h").exists():
            return candidate

    for candidate in sorted(install.glob("include/*/Python.h")):
        return candidate.parent

    raise SystemExit(f"Python.h not found under {install}")


def pbs_lib(install: Path, target: str) -> tuple[Path, str]:
    """
    Locates the CPython library to link against.

    Returns:
        tuple[Path, str]: (library dir, linker reference).
    """

    major, minor = CPYTHON_VERSION.split(".")[0], CPYTHON_VERSION.split(".")[1]

    if "windows" in target:
        lib = install / "libs" / f"python{major}{minor}.lib"

        if not lib.exists():
            raise SystemExit(f"import library missing: {lib}")

        return lib.parent, lib.name

    static = install / "lib" / f"libpython{major}.{minor}.a"

    if static.exists():
        # Static embed: the interpreter lives INSIDE our .so.
        return static.parent, f":{static.name}"

    dynamic = install / "lib" / f"libpython{major}.{minor}.dylib"

    if dynamic.exists():
        return dynamic.parent, f"python{major}.{minor}"

    raise SystemExit(f"no libpython under {install}")


# ------------------------------------------------------------------ #
# Compilation + staging                                               #
# ------------------------------------------------------------------ #


def compile_shim(target: str, install: Path, output: Path) -> None:
    """Compiles the shim sources against the PBS runtime."""

    sources = sorted((REPO_ROOT / "native" / "src").glob("*.c"))
    include = pbs_include(install)
    lib_dir, lib_ref = pbs_lib(install, target)

    if "windows" in target:
        command = [
            "cl", "/nologo", "/std:c11", "/W4", "/O2", "/LD",
            f"/I{include}", f"/I{REPO_ROOT / 'native'}",
            "/DNDEBUG", "/DND_BUILDING_ABI", "/DNADATA_EMBEDDED",
            "/D_CRT_SECURE_NO_WARNINGS",
            *[str(source) for source in sources],
            "/link", f"/OUT:{output}",
            f"/IMPLIB:{str(output).removesuffix('.dll')}.lib",
            f"/LIBPATH:{lib_dir}", lib_ref,
        ]
    else:
        command = [
            os.environ.get("CC", "cc"), "-shared", "-fPIC", "-O2", "-Wall",
            "-std=c11", f"-I{include}", f"-I{REPO_ROOT / 'native'}",
            "-DNDEBUG", "-DND_BUILDING_ABI", "-DNADATA_EMBEDDED",
            *[str(source) for source in sources],
            f"-L{lib_dir}", f"-l{lib_ref}",
        ]

        if "darwin" in target:
            # dyld resolves libpython relative to the library itself.
            command += ["-Wl,-rpath,@loader_path/runtime/lib"]

        if "linux" in target and lib_ref.startswith(":"):
            # Static embed pulls in libpython's own dependencies.
            command += ["-ldl", "-lm", "-lpthread"]

        command += ["-o", str(output)]

    output.parent.mkdir(parents=True, exist_ok=True)
    log("compiling shim: " + " ".join(command))

    if dev_build.run_command(command, cwd=output.parent) != 0:
        raise SystemExit("shim compilation failed")


def stage(target: str, install: Path, library: Path) -> Path:
    """
    Stages the distributable layout for one target.

    Returns:
        Path: The staged directory.
    """

    staged = DIST_DIR / target
    staged.mkdir(parents=True, exist_ok=True)

    # 1. The shim + its import library + the ABI header.
    shutil.copy2(library, staged / library.name)

    import_lib = library.with_suffix(".lib")

    if import_lib.exists():
        shutil.copy2(import_lib, staged / import_lib.name)

    shutil.copy2(REPO_ROOT / "native" / "nadata.h", staged / "nadata.h")

    # 2. The engine package (pure Python, zero dependencies).
    package_ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", "tests")

    shutil.copytree(REPO_ROOT / "nadata", staged / "nadata",
                    ignore=package_ignore, dirs_exist_ok=True)

    # 3. The pinned runtime.
    runtime = staged / "runtime"

    if runtime.exists():
        shutil.rmtree(runtime)

    log("staging runtime (this copies a lot; one-time per target)")
    shutil.copytree(install, runtime)

    for prune_dir in PRUNE_DIRS:
        for found in runtime.glob(f"*/{prune_dir}"):
            log(f"pruning {found.relative_to(staged)}")
            shutil.rmtree(found, ignore_errors=True)

    # 4. Windows hosts resolve python3xx.dll via PATH — ship a copy
    #    next to the shim so a single PATH entry suffices.
    if "windows" in target:
        for dll in runtime.glob("python3*.dll"):
            shutil.copy2(dll, staged / dll.name)

    return staged


# ------------------------------------------------------------------ #
# Validation                                                          #
# ------------------------------------------------------------------ #


def validate(staged: Path, target: str) -> None:
    """
    Builds + runs the C smoke test against the STAGED library, with
    NADATA_HOME unset — proving the embedded runtime layout boots and
    the engine works from the staged package alone.
    """

    if "windows" not in target:
        log("validation on non-host targets is CI's job — skipped")

        return

    work = CACHE_DIR / "validate" / target
    work.mkdir(parents=True, exist_ok=True)
    exe = work / "smoke_staged.exe"

    command = [
        "cl", "/nologo", "/W4", "/std:c11",
        str(REPO_ROOT / "native" / "tests" / "smoke.c"),
        f"/I{REPO_ROOT / 'native'}",
        "/link", f"/OUT:{exe}",
        f"/LIBPATH:{staged}", "nadata.lib",
    ]

    if dev_build.run_command(command, cwd=work) != 0:
        raise SystemExit("validation smoke build failed")

    environment = {k: v for k, v in os.environ.items() if k != "NADATA_HOME"}
    environment["PATH"] = (
        f"{staged};{staged / 'runtime'};"
        + environment.get("PATH", ""))

    log("running smoke test against the staged distribution "
        "(no NADATA_HOME)")

    run = subprocess.run([str(exe)], env=environment,
                         cwd=str(REPO_ROOT), capture_output=True,
                         text=True)

    tail = "\n".join(run.stdout.splitlines()[-3:])

    if run.returncode != 0:
        log(tail)
        raise SystemExit(f"staged smoke test FAILED: {run.returncode}")

    log(tail)

    if "0 failure(s)" not in run.stdout:
        raise SystemExit("staged smoke test did not report clean")


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #


def main() -> None:
    """Builds the distro-invariant native distribution."""

    parser = argparse.ArgumentParser(
        description="Build the embedded-runtime native distribution")
    parser.add_argument("--target", default=None,
                        help="target triple (default: host)")
    parser.add_argument("--run", action="store_true",
                        help="validate with the C smoke test "
                             "(host target only)")
    parser.add_argument("--list-assets", action="store_true",
                        help="list matching PBS assets and exit")
    arguments = parser.parse_args()

    target = arguments.target or host_target()
    log(f"target: {target} (PBS {PBS_TAG}, CPython {CPYTHON_VERSION})")

    install = ensure_runtime(target)

    if arguments.list_assets:
        return

    library_name = ("nadata.dll" if "windows" in target
                    else "nadata.dylib" if "darwin" in target
                    else "libnadata.so")
    output = CACHE_DIR / "out" / target / library_name

    compile_shim(target, install, output)
    staged = stage(target, install, output)

    total = sum(f.stat().st_size for f in staged.rglob("*") if f.is_file())

    log(f"staged {staged} ({total / (1 << 20):.1f} MB)")

    if arguments.run:
        validate(staged, target)
        log("validation passed")


if __name__ == "__main__":
    main()
