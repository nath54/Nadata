"""
Dev build for the native shim (Phase 1a).

Compiles native/src/*.c into a shared library (nadata.dll / 
libnadata.so / libnadata.dylib) linked against the HOST CPython — the
same interpreter that runs the test suite — and builds the C smoke
test against the result.

This is the development loop build. The distributable, distro-invariant
build (static libpython from python-build-standalone, Phase 1b) wraps
this same source; only the link step differs.

Usage:
    python native/build.py              # build dll + smoke test
    python native/build.py --run       # ... then run the smoke test

Prerequisites:
    - Windows: MSVC Build Tools (cl.exe), CPython headers/libs
    - Linux/macOS: any C compiler, python3-dev
    - The nadata package importable (pip install -e .) OR the
      NADATA_HOME env var pointing at it (dev path injection).
"""

# Import Modules
import argparse
import glob
import shutil
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

NATIVE_DIR: Path = Path(__file__).resolve().parent
SRC_DIR: Path = NATIVE_DIR / "src"
BUILD_DIR: Path = NATIVE_DIR / "build"
IS_WINDOWS: bool = os.name == "nt"


def find_vcvars() -> str | None:
    """
    Locates vcvars64.bat across common Visual Studio install roots.

    Returns:
        str | None: Path to vcvars64.bat, or None when MSVC is absent.
    """

    if not IS_WINDOWS:
        return None

    if shutil.which("cl") is not None:
        return None  # already on PATH; no wrapper needed

    patterns = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
    ]

    for pattern in patterns:
        matches = sorted(glob.glob(pattern))

        if matches:
            return matches[-1]  # newest edition/version

    return None


def run_command(command: list[str], cwd: Path) -> int:
    """
    Runs a build command, wrapping it in the MSVC environment when
    cl.exe is not already on PATH.

    cmd.exe strips the first and last quote of /c arguments (making
    quoted vcvars paths unusable inline), so the wrapped command is
    written to a temporary batch file instead — the only robust way.

    Args:
        command (list[str]): argv to execute.
        cwd (Path): Working directory for the build.

    Returns:
        int: Exit code.
    """

    vcvars = find_vcvars()

    if vcvars is None:
        result = subprocess.run(command, cwd=cwd)

        return result.returncode

    batch = cwd / "_msvc_build.cmd"

    batch.write_text(
        "@echo off\r\n"
        f"call \"{vcvars}\" >nul 2>&1\r\n"
        f"{subprocess.list2cmdline(command)}\r\n",
        encoding="utf-8",
    )

    try:
        result = subprocess.run(["cmd", "/c", str(batch)], cwd=cwd)

        return result.returncode
    finally:
        batch.unlink(missing_ok=True)


def python_include() -> str:
    """
    Returns the CPython include directory.

    Returns:
        str: Include path (validated to contain Python.h).
    """

    include = sysconfig.get_paths()["include"]

    if not (Path(include) / "Python.h").exists():
        raise SystemExit(
            f"Python.h not found in {include} — install the dev headers "
            "(python3-dev / the full CPython installer)"
        )

    return include


def python_library() -> tuple[str, str]:
    """
    Locates the CPython import library.

    Returns:
        tuple[str, str]: (directory, file name) of the import lib.
    """

    if IS_WINDOWS:
        libs_dir = Path(sys.base_prefix) / "libs"
        # Windows venvs point the include dir at the base install.
        candidates = [
            Path(sysconfig.get_config_var("LIBDIR") or ""),
            libs_dir,
        ]

        win_suffix = f"{sys.version_info.major}{sys.version_info.minor}"

        for candidate in candidates:
            lib = candidate / f"python{win_suffix}.lib"

            if lib.exists():
                return str(lib.parent), lib.name

        raise SystemExit(
            f"python{win_suffix}.lib not found — install the full CPython"
        )

    libdir = sysconfig.get_config_var("LIBDIR") or ""
    name = sysconfig.get_config_var("LDLIBRARY") or ""

    # macOS framework builds report LDLIBRARY as a path INSIDE the
    # framework ("Python.framework/Versions/3.12/Python") which no -l
    # flag can consume — the framework also ships lib/libpythonX.Y.dylib
    # in LIBDIR, so fall through to the glob below.
    if "/" in name:
        name = ""

    # Some installs (python.org / setup-python builds) report
    # LDLIBRARY but ship only the versioned file — glob a fallback.
    if not name or name.startswith("LDLIBRARY-NOT-FOUND"):
        major_minor = suffix()

        # Framework installs keep the dylib under .../Versions/X.Y/lib;
        # also probe the interpreter's own prefix for completeness.
        search_dirs = [d for d in (libdir, sys.base_prefix) if d]

        for directory in search_dirs:
            for pattern in (f"libpython{major_minor}.*",
                            f"libpython{major_minor.replace('.', '')}.*"):
                matches = sorted(Path(directory).glob(pattern))

                if matches:
                    return str(matches[-1].parent), matches[-1].name

        raise SystemExit(
            f"no libpython{major_minor} found under "
            f"{', '.join(search_dirs)} "
            "— install the dev headers/libraries (python3-dev)")

    return libdir, name


def suffix() -> str:
    """
    Returns the Unix library suffix for the running interpreter.

    Returns:
        str: e.g. "3.12".
    """

    return f"{sys.version_info.major}.{sys.version_info.minor}"


def compiler_command(target: str, sources: list[Path], include: str,
                     lib_dir: str, lib_name: str) -> list[str]:
    """
    Builds the compiler/linker command line for the host toolchain.

    Args:
        target (str): Output file path.
        sources (list[Path]): C source files.
        include (str): CPython include dir.
        lib_dir (str): CPython import library dir.
        lib_name (str): CPython import library file name.

    Returns:
        list[str]: argv for subprocess.run.
    """

    if IS_WINDOWS:
        return [
            "cl", "/nologo", "/std:c11", "/W4", "/O2", "/LD",
            f"/I{include}", f"/I{NATIVE_DIR}", "/DNDEBUG",
            "/DND_BUILDING_ABI", "/D_CRT_SECURE_NO_WARNINGS",
            *[str(source) for source in sources],
            "/link", f"/OUT:{target}",
            f"/IMPLIB:{str(target).removesuffix('.dll')}.lib",
            f"/LIBPATH:{lib_dir}", lib_name,
        ]

    return [
        os.environ.get("CC", "cc"), "-shared", "-fPIC", "-O2", "-Wall",
        "-std=c11", f"-I{include}", f"-I{NATIVE_DIR}", "-DNDEBUG",
        *[str(source) for source in sources],
        f"-L{lib_dir}", f"-l{lib_name_lib(lib_name)}",
        *(["-Wl,-install_name,@rpath/libnadata.dylib"]
          if sys.platform == "darwin" else []),
        "-o", target,
    ]


def lib_name_lib(name: str) -> str:
    """
    Converts a linker library file name to the -l argument.

    Args:
        name (str): Library file name (e.g. libpython3.12.so).

    Returns:
        str: The -l argument (e.g. python3.12).
    """

    if name.startswith("lib"):
        # Strip the lib prefix and any .so / .dylib extension — -l
        # resolves libpython3.12.dylib from just -lpython3.12.
        stem = name[3:]

        if stem.endswith((".so", ".dylib")):
            stem = stem.rsplit(".", 1)[0]

        return stem

    return name


def build() -> Path:
    """
    Compiles the shim and the smoke test.

    Returns:
        Path: The built shared library.
    """

    include = python_include()
    lib_dir, lib_name = python_library()
    sources = sorted(SRC_DIR.glob("*.c"))

    if not sources:
        raise SystemExit(f"no sources found in {SRC_DIR}")

    build_dir = BUILD_DIR / sys.platform
    build_dir.mkdir(parents=True, exist_ok=True)

    # POSIX linkers only resolve -lnadata against libnadata.*; macOS
    # additionally uses the .dylib suffix.
    if IS_WINDOWS:
        library = build_dir / "nadata.dll"
    elif sys.platform == "darwin":
        library = build_dir / "libnadata.dylib"
    else:
        library = build_dir / "libnadata.so"

    command = compiler_command(str(library), sources, include, lib_dir,
                               lib_name)

    print("[build]", " ".join(command))

    if run_command(command, cwd=build_dir) != 0:
        raise SystemExit("build failed")

    print(f"[build] wrote {library}")

    return library


def run_smoke(library: Path) -> None:
    """
    Runs the smoke test against the built library.

    Args:
        library (Path): The built shared library.
    """

    build_dir = library.parent
    environment = os.environ.copy()

    # Ensure `import nadata` resolves (dev path injection in the shim).
    if "NADATA_HOME" not in environment:
        environment["NADATA_HOME"] = str(Path.cwd())

    if IS_WINDOWS:
        # cl.exe writes the import lib next to /OUT; smoke.exe needs
        # the dll + python dll reachable via PATH.
        environment["PATH"] = (
            f"{build_dir};{Path(sys.base_prefix)};"
            f"{Path(sys.base_prefix) / 'DLLs'};"
            + environment.get("PATH", "")
        )
        import_lib = build_dir / "nadata.lib"
        exe = build_dir / "smoke.exe"

        link = [
            "cl", "/nologo", "/W4", "/std:c11",
            str(NATIVE_DIR / "tests" / "smoke.c"),
            f"/I{NATIVE_DIR}",
            "/link", f"/OUT:{exe}",
            f"/LIBPATH:{build_dir}", "nadata.lib",
        ]

        if not import_lib.exists():
            raise SystemExit(f"missing import lib: {import_lib}")
    else:
        exe = build_dir / "smoke"
        link = [
            os.environ.get("CC", "cc"), "-O2", "-Wall", "-std=c11",
            str(NATIVE_DIR / "tests" / "smoke.c"),
            f"-I{NATIVE_DIR}", "-o", str(exe),
            f"-L{build_dir}", "-lnadata",
            f"-Wl,-rpath,{build_dir}",
        ]

    print("[smoke]", " ".join(link))

    if run_command(link, cwd=build_dir) != 0:
        raise SystemExit("smoke build failed")

    run = subprocess.run([str(exe)], env=environment,
                         cwd=str(Path.cwd().resolve()))

    if run.returncode != 0:
        raise SystemExit(f"smoke test FAILED: {run.returncode}")

    print("[smoke] all cases passed")


def main() -> None:
    """
    Parses arguments, builds, optionally runs the smoke test.
    """

    parser = argparse.ArgumentParser(description="Build the nadata shim")
    parser.add_argument("--run", action="store_true",
                        help="run the C smoke test after building")
    arguments = parser.parse_args()

    library = build()

    if arguments.run:
        run_smoke(library)


if __name__ == "__main__":
    main()
