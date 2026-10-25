"""
Phase 1b — distro-invariant native distribution builder.

Produces a self-contained native release for one target platform:

    dist/native/<target[-variant]>/
        nadata.dll | libnadata.so | libnadata.dylib  <- the shim
        nadata.h                                   <- the ABI header
        nadata/                                    <- the engine package
        runtime/                                   <- pinned CPython
        python312.dll (Windows copy, next to the shim)

The interpreter comes from python-build-standalone (PBS) — the same
runtimes `uv` uses — verified against the release SHA256SUMS. On Linux
the STATIC libpython is linked into libnadata.so itself (true distro
invariance); on Windows/macOS the shared runtime ships alongside and
`PyConfig.home` is set relative to the library location by the shim
(shim_runtime.c boot_interpreter).

The default build is deterministic: PBS 20241016 / CPython 3.12.7.
It can be retargeted per invocation:

    --python 3.14        resolve the newest 3.14.x across PBS releases
    --pbs-tag 20250108   pin a specific PBS release
    --free-threaded      the nogil build (3.13+; -ft suffix)
    --jit                a JIT-enabled build if PBS ships one (rare)

Usage:
    python scripts/build_native.py                  # deterministic pin
    python scripts/build_native.py --python 3.14 --run
    python scripts/build_native.py --list-assets    # inspect resolution

The result is validated by running the C smoke test against the STAGED
library with NADATA_HOME unset (proving the embedded layout boots).
"""

# Import Modules
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "native"))

# pylint: disable=wrong-import-position
import build as dev_build  # native/build.py helpers (vcvars, run_command)

# Deterministic default pin (see the 3.12 rationale in git history).
PBS_REPO = "astral-sh/python-build-standalone"
DEFAULT_PBS_TAG = "20241016"
DEFAULT_PYTHON = "3.12.7"
MAX_RELEASE_SCAN = 10  # releases walked when resolving a --python spec

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
# PBS release + asset resolution                                      #
# ------------------------------------------------------------------ #


API_CACHE = CACHE_DIR / "api"
API_CACHE_TTL = 24 * 3600  # seconds


def fetch_json(url: str):
    """GETs a JSON document (GitHub API), cached for a day.

    The unauthenticated GitHub API allows 60 requests/hour; resolution
    walks several releases, so responses are cached on disk.
    """

    API_CACHE.mkdir(parents=True, exist_ok=True)

    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    cache = API_CACHE / f"{key}.json"

    if cache.exists() and time.time() - cache.stat().st_mtime < API_CACHE_TTL:
        return json.loads(cache.read_text(encoding="utf-8"))

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 403:
            raise SystemExit(
                "GitHub API rate limit exceeded (60/h unauthenticated). "
                "Retry later, or pass --pbs-tag with an archive already "
                "in .native-cache/downloads for a fully offline build."
            ) from error

        raise

    cache.write_text(json.dumps(payload), encoding="utf-8")

    return payload


def fetch_assets(tag: str) -> list[dict]:
    """Lists the assets of one PBS release via the GitHub API."""

    return fetch_json(
        f"https://api.github.com/repos/{PBS_REPO}/releases/tags/{tag}"
    ).get("assets", [])


def list_release_tags() -> list[str]:
    """Release tags newest-first (PBS tags are date-based)."""

    releases = fetch_json(
        f"https://api.github.com/repos/{PBS_REPO}/releases"
        f"?per_page={MAX_RELEASE_SCAN}"
    )

    return [release["tag_name"] for release in releases]


def parse_cpython_version(name: str) -> str | None:
    """Extracts 'cpython-<version>+' from an asset name (None otherwise)."""

    if not name.startswith("cpython-"):
        return None

    version = name.split("+")[0].removeprefix("cpython-")

    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        return None  # skips pre-releases like 3.14.0a1

    return version


def version_key(version: str) -> tuple[int, ...]:
    """Numeric sort/compare key for an exact CPython version."""

    return tuple(int(part) for part in version.split("."))


def asset_matches(name: str, triple: str, free_threaded: bool,
                  jit: bool) -> bool:
    """True when an asset name fits the target + build variant."""

    return (name.startswith("cpython-")
            and triple in name
            and "install_only" in name
            and "stripped" not in name
            and free_threaded == ("freethreaded" in name)
            and jit == ("jit" in name))


def candidate_assets(assets: list[dict], triple: str, free_threaded: bool,
                     jit: bool) -> list[dict]:
    """
    Filters install-only archives for a triple and build variant.

    Asset names look like:
      cpython-3.12.7+20241016-x86_64-pc-windows-msvc-install_only.tar.gz
      cpython-3.13.1+20250108-x86_64-...-freethreaded+shared-install_only
    """

    matches = []

    for asset in assets:
        name = asset["name"]

        if asset_matches(name, triple, free_threaded, jit):
            matches.append(asset)

    # Plainest name first (avoids +debug/+pgo variants when present).
    matches.sort(key=lambda asset: (len(asset["name"]), asset["name"]))

    return matches


def resolve_build(python_spec: str, triple: str, free_threaded: bool,
                  jit: bool, tag_hint: str | None) -> tuple[str, str]:
    """
    Resolves a --python spec to a (PBS tag, exact CPython version).

    Walks the newest releases first; a tag_hint pins the search to one
    release. The spec may be "3.12" (newest patch) or "3.12.7" (exact).
    """

    wanted = tuple(int(part) for part in python_spec.split("."))
    tags = [tag_hint] if tag_hint else list_release_tags()

    # Offline fast path: a previously downloaded archive for this
    # pin is authoritative (it gets checksum-verified at extraction).
    if tag_hint:
        for path in sorted(
                (CACHE_DIR / "downloads").glob(
                    f"cpython-*+{tag_hint}-{triple}*install_only.tar.gz")):
            if not asset_matches(path.name, triple, free_threaded, jit):
                continue

            version = parse_cpython_version(path.name)

            if version is not None \
                    and version_key(version)[:len(wanted)] == wanted:
                return tag_hint, version

    for tag in tags:
        assets = fetch_assets(tag)

        for asset in candidate_assets(assets, triple, free_threaded, jit):
            version = parse_cpython_version(asset["name"])

            if version is None:
                continue

            if version_key(version)[:len(wanted)] == wanted:
                return tag, version

    scanned = ", ".join(tags[:3]) + ("..." if len(tags) > 3 else "")

    if jit:
        raise SystemExit(
            f"no JIT-enabled PBS build for CPython {python_spec} "
            f"({scanned}). PBS ships JIT builds rarely; a JIT runtime "
            "needs building CPython from source (--from-source, planned)")

    if free_threaded:
        raise SystemExit(
            f"no free-threaded PBS build for CPython {python_spec} "
            f"({scanned}) — free-threading requires 3.13+")

    raise SystemExit(
        f"no PBS build for CPython {python_spec} in {scanned}")


# ------------------------------------------------------------------ #
# Download + verification                                             #
# ------------------------------------------------------------------ #


def sha256_of(path: Path) -> str:
    """Streams a file through SHA256."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_checksum(downloads: Path, name: str, sums_url: str | None,
                    tag: str) -> None:
    """Checks a downloaded archive against the release SHA256SUMS."""

    sums_path = downloads / f"SHA256SUMS-{tag}"

    if not sums_path.exists():
        if sums_url is None:
            raise SystemExit(
                f"SHA256SUMS for {tag} not cached and no API access")

        urllib.request.urlretrieve(sums_url, sums_path)

    expected = None

    for line in sums_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if len(parts) == 2 and parts[1] == name:
            expected = parts[0]

            break

    if expected is None:
        raise SystemExit(f"{name} not listed in SHA256SUMS")

    actual = sha256_of(downloads / name)

    if actual != expected:
        raise SystemExit(f"checksum mismatch for {name}: {actual}")

    log(f"checksum ok: {name}")


def find_downloaded_archive(version: str, tag: str, triple: str,
                            free_threaded: bool, jit: bool) -> Path | None:
    """Locates a previously downloaded archive for this pin (or None)."""

    for path in sorted(
            (CACHE_DIR / "downloads").glob(
                f"cpython-{version}+{tag}-{triple}*install_only.tar.gz")):
        if asset_matches(path.name, triple, free_threaded, jit):
            return path

    return None


def extract_runtime(extract_dir: Path, archive_path: Path) -> Path:
    """Extracts an install_only archive (idempotent) -> install dir."""

    if not (extract_dir / "python").exists():
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


def ensure_runtime(tag: str, version: str, triple: str,
                   free_threaded: bool, jit: bool) -> Path:
    """
    Downloads, verifies and extracts the pinned PBS runtime.

    Returns:
        Path: The extracted install directory (contains include/, lib/).
    """

    downloads = CACHE_DIR / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    # Offline fast path: archive + sums already on disk -> no API call.
    local = find_downloaded_archive(version, tag, triple, free_threaded,
                                    jit)
    sums_local = downloads / f"SHA256SUMS-{tag}"

    extract_dir = CACHE_DIR / "pbs" / f"{version}-{triple}"

    if free_threaded:
        extract_dir = CACHE_DIR / "pbs" / f"{version}t-{triple}"

    if jit:
        extract_dir = extract_dir.with_name(extract_dir.name + "-jit")

    if local is not None and sums_local.exists():
        verify_checksum(downloads, local.name, None, tag)

        return extract_runtime(extract_dir, local)

    assets = fetch_assets(tag)
    archive = next(
        (a for a in candidate_assets(assets, triple, free_threaded, jit)
         if a["name"].startswith(f"cpython-{version}+")),
        None,
    )

    if archive is None:
        if local is not None:
            # API unavailable, but we hold the archive: use it.
            return extract_runtime(extract_dir, local)

        raise SystemExit(
            f"asset for {version} disappeared from release {tag}")

    sums = next(
        (a for a in assets if a["name"] == "SHA256SUMS"), None)

    if sums is None:
        raise SystemExit(f"SHA256SUMS asset missing from {tag}")

    name = archive["name"]
    archive_path = downloads / name

    if not archive_path.exists():
        log(f"downloading {name}")
        urllib.request.urlretrieve(archive["browser_download_url"],
                                   archive_path)

    verify_checksum(downloads, name, sums["browser_download_url"], tag)

    return extract_runtime(extract_dir, archive_path)


# ------------------------------------------------------------------ #
# Layout discovery inside the PBS install                             #
# ------------------------------------------------------------------ #


def pbs_include(install: Path) -> Path:
    """Locates the directory containing Python.h."""

    if (install / "include" / "Python.h").exists():
        return install / "include"

    for candidate in sorted(install.glob("include/*/Python.h")):
        return candidate.parent

    raise SystemExit(f"Python.h not found under {install}")


def pbs_lib(install: Path, target: str, free_threaded: bool) -> tuple[Path, str]:
    """
    Locates the CPython library to link against.

    Returns:
        tuple[Path, str]: (library dir, linker reference).
    """

    version = pbs_version(install)
    major, minor = version.split(".")[:2]
    suffix = "t" if free_threaded else ""

    if "windows" in target:
        lib = install / "libs" / f"python{major}{minor}{suffix}.lib"

        if not lib.exists():
            raise SystemExit(f"import library missing: {lib}")

        return lib.parent, lib.name

    static = install / "lib" / f"libpython{major}.{minor}{suffix}.a"

    if static.exists():
        # Static embed: the interpreter lives INSIDE our .so.
        return static.parent, f":{static.name}"

    dynamic = install / "lib" / f"libpython{major}.{minor}{suffix}.dylib"

    if dynamic.exists():
        return dynamic.parent, f"python{major}.{minor}{suffix}"

    # Linux shared builds ship the versioned SONAME (and sometimes an
    # unversioned dev symlink) — link with the exact file via -l:NAME.
    plain = install / "lib" / f"libpython{major}.{minor}{suffix}.so"

    if plain.exists():
        return plain.parent, f"python{major}.{minor}{suffix}"

    versioned = install / "lib" / f"libpython{major}.{minor}{suffix}.so.1.0"

    if versioned.exists():
        return versioned.parent, f":{versioned.name}"

    raise SystemExit(f"no libpython under {install}")


def pbs_version(install: Path) -> str:
    """Reads the CPython version from the runtime's include dir."""

    for header in sorted(install.glob("include/*/patchlevel.h")) + [
            install / "include" / "patchlevel.h"]:
        for line in header.read_text(encoding="utf-8").splitlines():
            if line.startswith("#define PY_VERSION"):
                return line.split('"')[1]

    raise SystemExit(f"cannot determine CPython version under {install}")


# ------------------------------------------------------------------ #
# Compilation + staging                                               #
# ------------------------------------------------------------------ #


def compile_shim(target: str, install: Path, output: Path,
                 free_threaded: bool) -> None:
    """Compiles the shim sources against the PBS runtime."""

    sources = sorted((REPO_ROOT / "native" / "src").glob("*.c"))
    include = pbs_include(install)
    lib_dir, lib_ref = pbs_lib(install, target, free_threaded)

    if "darwin" in target:
        # PBS records the dylib's install name as an ABSOLUTE build-
        # machine path (/install/lib/...); dyld will not consult rpath
        # for absolute dependencies. Rewrite it to @rpath (resolved by
        # the shim's and hosts' @loader_path/runtime/lib rpaths) and
        # re-sign ad hoc, since the edit invalidates the signature.
        version = pbs_version(install)
        major, minor = version.split(".")[:2]
        suffix = "t" if free_threaded else ""
        dylib = (install / "lib"
                 / f"libpython{major}.{minor}{suffix}.dylib")
        subprocess.run(
            ["install_name_tool", "-id",
             f"@rpath/libpython{major}.{minor}{suffix}.dylib", str(dylib)],
            check=True)
        subprocess.run(["codesign", "--force", "--sign", "-",
                        str(dylib)], check=True)

    # Free-threaded builds must be compiled with Py_GIL_DISABLED; it
    # also flips nd_runtime_info's free_threaded field.
    gil_flag = ["-DPy_GIL_DISABLED=1"] if free_threaded else []

    if "windows" in target:
        command = [
            "cl", "/nologo", "/std:c11", "/W4", "/O2", "/LD",
            f"/I{include}", f"/I{REPO_ROOT / 'native'}",
            "/DNDEBUG", "/DND_BUILDING_ABI", "/DNADATA_EMBEDDED",
            "/D_CRT_SECURE_NO_WARNINGS",
            *[flag.replace("-D", "/D") for flag in gil_flag],
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
            *gil_flag,
            *[str(source) for source in sources],
            f"-L{lib_dir}", f"-l{lib_ref}",
        ]

        if "darwin" in target:
            # dyld resolves libpython relative to the library itself.
            command += ["-Wl,-rpath,@loader_path/runtime/lib"]

        if "linux" in target:
            # Same idea with the ELF token: the staged layout keeps
            # libpython under runtime/lib next to the shim.
            command += ["-Wl,-rpath,$ORIGIN/runtime/lib"]

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
        # Windows stdlib: <runtime>/Lib/<dir>; POSIX: <runtime>/lib/
        # python3.X/<dir>.
        for pattern in (f"*/{prune_dir}",
                        f"*/lib/python*/{prune_dir}",
                        f"*/Lib/{prune_dir}"):
            for found in runtime.glob(pattern):
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

    work = CACHE_DIR / "validate" / target
    work.mkdir(parents=True, exist_ok=True)

    exe_name = "smoke_staged.exe" if "windows" in target \
        else "smoke_staged"
    exe = work / exe_name

    if "windows" in target:
        command = [
            "cl", "/nologo", "/W4", "/std:c11",
            str(REPO_ROOT / "native" / "tests" / "smoke.c"),
            f"/I{REPO_ROOT / 'native'}",
            "/link", f"/OUT:{exe}",
            f"/LIBPATH:{staged}", "nadata.lib",
        ]
    else:
        # rpath points at the staged dir: on Linux the interpreter is
        # statically inside libnadata.so, on macOS the dylib resolves
        # via the runtime/ rpath compiled into the shim. The extra exe
        # rpath also covers libpython's own dependencies on macOS.
        command = [
            os.environ.get("CC", "cc"), "-O2", "-Wall", "-std=c11",
            str(REPO_ROOT / "native" / "tests" / "smoke.c"),
            f"-I{REPO_ROOT / 'native'}",
            f"-L{staged}", "-lnadata",
            f"-Wl,-rpath,{staged}",
            f"-Wl,-rpath,{staged / 'runtime' / 'lib'}",
            "-o", str(exe),
        ]

    if dev_build.run_command(command, cwd=work) != 0:
        raise SystemExit("validation smoke build failed")

    # Sanitize the environment: no NADATA_HOME (proves the staged
    # layout self-locates) and no host-python library paths —
    # LD_LIBRARY_PATH from the runner would make the loader resolve
    # libpython3.12.so.1.0 to the HOST interpreter (e.g. 3.12.15)
    # while the stdlib is the pinned 3.12.7 -> segfault at boot.
    environment = {
        key: value for key, value in os.environ.items()
        if not (key == "NADATA_HOME"
                or key.startswith("DYLD_")
                or key in ("LD_LIBRARY_PATH", "PYTHONHOME",
                           "PYTHONPATH", "PYTHONSTARTUP"))
    }

    if "windows" in target:
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
        log("--- smoke stdout (tail) ---")
        log(tail or "(empty)")
        log("--- smoke stderr (tail) ---")
        log("\n".join(run.stderr.splitlines()[-10:]) or "(empty)")
        raise SystemExit(f"staged smoke test FAILED: {run.returncode}")

    log(tail)

    if "0 failure(s)" not in run.stdout:
        raise SystemExit("staged smoke test did not report clean")


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #


def variant_suffix(free_threaded: bool, jit: bool) -> str:
    """Distinguishes staged variants of the same target triple."""

    suffix = ""

    if free_threaded:
        suffix += "-freethreaded"

    if jit:
        suffix += "-jit"

    return suffix


def main() -> None:
    """Builds the distro-invariant native distribution."""

    parser = argparse.ArgumentParser(
        description="Build the embedded-runtime native distribution")
    parser.add_argument("--target", default=None,
                        help="target triple (default: host)")
    parser.add_argument("--python", default=DEFAULT_PYTHON,
                        help="CPython version: '3.12' (newest patch) or "
                             f"'3.12.7' (exact); default {DEFAULT_PYTHON}")
    parser.add_argument("--pbs-tag", default=None,
                        help="pin one PBS release (default: newest "
                             "release carrying the requested version)")
    parser.add_argument("--free-threaded", action="store_true",
                        help="use the nogil build (CPython 3.13+)")
    parser.add_argument("--jit", action="store_true",
                        help="use a JIT-enabled build (rarely shipped "
                             "by PBS; needs --from-source otherwise)")
    parser.add_argument("--run", action="store_true",
                        help="validate with the C smoke test "
                             "(host target only)")
    parser.add_argument("--list-assets", action="store_true",
                        help="resolve the build and list matching "
                             "assets, then exit")
    arguments = parser.parse_args()

    base_target = arguments.target or host_target()
    target = base_target + variant_suffix(
        arguments.free_threaded, arguments.jit)

    # The deterministic default stays pinned to its release; explicit
    # --python specs resolve against the newest releases instead.
    tag_hint = arguments.pbs_tag

    if tag_hint is None and arguments.python == DEFAULT_PYTHON:
        tag_hint = DEFAULT_PBS_TAG

    pbs_tag, version = resolve_build(
        arguments.python, base_target,
        arguments.free_threaded, arguments.jit, tag_hint)

    log(f"target: {target} (PBS {pbs_tag}, CPython {version}, "
        f"free-threaded={arguments.free_threaded}, jit={arguments.jit})")

    install = ensure_runtime(pbs_tag, version, base_target,
                             arguments.free_threaded, arguments.jit)

    if arguments.list_assets:
        return

    library_name = ("nadata.dll" if "windows" in base_target
                    else "libnadata.dylib" if "darwin" in base_target
                    else "libnadata.so")
    output = CACHE_DIR / "out" / target / library_name

    compile_shim(base_target, install, output, arguments.free_threaded)
    staged = stage(target, install, output)

    total = sum(f.stat().st_size for f in staged.rglob("*") if f.is_file())

    log(f"staged {staged} ({total / (1 << 20):.1f} MB)")

    if arguments.run:
        validate(staged, base_target)
        log("validation passed")


if __name__ == "__main__":
    main()
