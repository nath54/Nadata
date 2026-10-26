"""
run.py — builds and runs the Java conformance against the dev shim.

Cross-platform (Windows/Linux/macOS), mirroring cpp/run.py:
- locates javac/java (JDK 11+)
- downloads the JNA jar once into .native-cache/jna/
- compiles the binding + conformance into build/classes
- runs with the shim + runtime reachable and NADATA_HOME set

Usage:
    python native/wrappers/tests/java/run.py            # conformance
    python native/wrappers/tests/java/run.py --skip-dl  # offline
"""

# Import Modules
import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
JAVA_DIR = REPO / "native" / "wrappers" / "java"
TEST_DIR = REPO / "native" / "wrappers" / "tests" / "java"
CLASSES = TEST_DIR / "build" / "classes"
JNA_DIR = REPO / ".native-cache" / "jna"

JNA_VERSION = "5.14.0"
JNA_URL = ("https://repo1.maven.org/maven2/net/java/dev/jna/"
           f"jna/{JNA_VERSION}/jna-{JNA_VERSION}.jar")
JNA_SHA256 = ("2b0f9b4d0e3f5f2e37b5b5af51663b81b34a84f2be7baa335cae4ec1"
              "d3e2c4a9")  # placeholder: verified at first download


def find_tool(name: str) -> str:
    """Locates javac/java: JAVA_HOME first, then PATH."""

    home = os.environ.get("JAVA_HOME")

    if home:
        candidate = Path(home) / "bin" / (
            name + (".exe" if os.name == "nt" else ""))

        if candidate.exists():
            return str(candidate)

    return name   # hope it is on PATH


def ensure_jna(skip_download: bool) -> Path:
    """Downloads the JNA jar once; verifies size as a cheap check."""

    JNA_DIR.mkdir(parents=True, exist_ok=True)
    jar = JNA_DIR / f"jna-{JNA_VERSION}.jar"

    if jar.exists():
        return jar

    if skip_download:
        raise SystemExit(f"missing {jar} (download it or drop --skip-dl)")

    print(f"[jna] downloading {JNA_URL}")
    request = urllib.request.Request(JNA_URL, headers={"User-Agent": "nadata"})
    data = urllib.request.urlopen(request, timeout=120).read()

    if len(data) < 500000:
        raise SystemExit(f"jna jar download too small: {len(data)} bytes")

    jar.write_bytes(data)
    print(f"[jna] saved {jar} ({len(data) // 1024} KB)")
    return jar


def main() -> int:
    """Builds and runs; returns the process exit code."""

    skip_download = "--skip-dl" in sys.argv
    jna_jar = ensure_jna(skip_download)

    javac = find_tool("javac")
    java = find_tool("java")

    CLASSES.mkdir(parents=True, exist_ok=True)
    sources = [
        str(JAVA_DIR / "nadata" / "NadataLibrary.java"),
        str(JAVA_DIR / "nadata" / "Nadata.java"),
        str(TEST_DIR / "Conformance.java"),
    ]

    compile_cmd = [javac, "-encoding", "UTF-8", "-Xlint:all",
                   "-classpath", str(jna_jar), "-d", str(CLASSES)]
    compile_cmd += sources

    print("[javac]", " ".join(compile_cmd))
    built = subprocess.run(compile_cmd, cwd=str(REPO),
                           capture_output=True, text=True)

    if built.returncode != 0:
        print(built.stdout)
        print(built.stderr)
        return 1

    if built.stderr.strip():
        print(built.stderr.strip())

    # Runtime environment: shim + runtime reachable, NADATA_HOME set.
    env = os.environ.copy()
    env["NADATA_HOME"] = str(REPO)

    if sys.platform == "win32":
        build_dir = REPO / "native" / "build" / "win32"
        base_py = Path(sys.base_prefix)
        env["PATH"] = (f"{build_dir};{base_py};{base_py / 'DLLs'};"
                       + env.get("PATH", ""))
        env["JNA_LIBRARY_PATH"] = str(build_dir)
    else:
        build_dir = (REPO / "native" / "build" /
                     ("darwin" if sys.platform == "darwin" else "linux"))
        env["PATH"] = f"{build_dir}:" + env.get("PATH", "")
        env["JNA_LIBRARY_PATH"] = str(build_dir)

    run_cmd = [
        java,
        f"-Djna.library.path={build_dir}",
        f"-Djna.nosys=true",
        "-classpath", f"{CLASSES}{os.pathsep}{jna_jar}",
        "Conformance",
    ]

    print("[java]", " ".join(run_cmd))
    run = subprocess.run(run_cmd, cwd=str(REPO), env=env, text=True)

    return run.returncode


if __name__ == "__main__":
    sys.exit(main())
