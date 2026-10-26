"""
run.py — builds and runs the Kotlin conformance against the dev shim.

Cross-platform (Windows/Linux/macOS), reusing the Java suite helpers:
- locates javac/java (JDK 11+)
- ensures the JNA jar in .native-cache/jna/ (shared with java/run.py)
- downloads the Kotlin compiler once into .native-cache/kotlinc/
- compiles the Java binding + Kotlin layer + conformance
- runs with the shim + runtime reachable and NADATA_HOME set

Usage:
    python native/wrappers/tests/kotlin/run.py            # conformance
    python native/wrappers/tests/kotlin/run.py --skip-dl  # offline
"""

# Import Modules
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
KOTLIN_DIR = REPO / "native" / "wrappers" / "kotlin"
TEST_DIR = REPO / "native" / "wrappers" / "tests" / "kotlin"
BUILD = TEST_DIR / "build"
JAVA_CLASSES = BUILD / "java"
KT_CLASSES = BUILD / "kotlin"

sys.path.insert(0, str(REPO / "native" / "wrappers" / "tests" / "java"))
import run as java_run  # noqa: E402  (find_tool, ensure_jna)

KOTLIN_VERSION = "2.0.21"
KOTLINC_URL = ("https://github.com/JetBrains/kotlin/releases/download/"
               f"v{KOTLIN_VERSION}/kotlin-compiler-{KOTLIN_VERSION}.zip")
KOTLINC_ROOT = REPO / ".native-cache" / "kotlinc"


def kotlinc_home() -> Path:
    """Locates (or downloads) the Kotlin compiler; returns its root."""

    dist = KOTLINC_ROOT / "kotlinc"
    marker = dist / "lib" / "kotlin-compiler.jar"

    if marker.exists():
        return dist

    if "--skip-dl" in sys.argv:
        raise SystemExit(f"missing {exe} (run without --skip-dl)")

    zip_path = KOTLINC_ROOT / f"kotlin-compiler-{KOTLIN_VERSION}.zip"
    KOTLINC_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[kotlinc] downloading {KOTLINC_URL}")
    request = urllib.request.Request(KOTLINC_URL,
                                     headers={"User-Agent": "nadata"})
    zip_path.write_bytes(
        urllib.request.urlopen(request, timeout=600).read())

    print(f"[kotlinc] extracting to {KOTLINC_ROOT}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(KOTLINC_ROOT)

    if not marker.exists():
        raise SystemExit(f"extracted but {marker} not found")

    return dist


def main() -> int:
    """Builds and runs; returns the process exit code."""

    jna_jar = java_run.ensure_jna("--skip-dl" in sys.argv)
    javac = java_run.find_tool("javac")

    # 1. Compile the Java binding (the Kotlin layer extends it).
    JAVA_CLASSES.mkdir(parents=True, exist_ok=True)

    compile_java = [
        javac, "-encoding", "UTF-8", "-classpath", str(jna_jar),
        "-d", str(JAVA_CLASSES),
        str(KOTLIN_DIR / ".." / "java" / "nadata" / "Nadata.java"),
        str(KOTLIN_DIR / ".." / "java" / "nadata" / "NadataLibrary.java"),
    ]

    print("[javac]", " ".join(compile_java))
    built = subprocess.run(compile_java, cwd=str(REPO),
                           capture_output=True, text=True)

    if built.returncode != 0:
        print(built.stdout)
        print(built.stderr)
        return 1

    # 2. Compile the Kotlin layer + conformance.
    # The kotlinc launcher scripts mishandle quoted paths containing
    # spaces (e.g. "Cerisara Nathan"), so drive the compiler jar
    # directly with the host JVM.
    dist = kotlinc_home()
    KT_CLASSES.mkdir(parents=True, exist_ok=True)
    java_classpath = f"{jna_jar}{os.pathsep}{JAVA_CLASSES}"

    compile_kt = [
        java_run.find_tool("java"),
        "-classpath", str(dist / "lib" / "kotlin-compiler.jar"),
        "org.jetbrains.kotlin.cli.jvm.K2JVMCompiler",
        "-classpath", java_classpath,
        "-jvm-target", "11",
        "-d", str(KT_CLASSES),
        str(KOTLIN_DIR / "nadata" / "NadataExt.kt"),
        str(TEST_DIR / "Conformance.kt"),
    ]

    print("[kotlinc]", " ".join(compile_kt))
    built = subprocess.run(compile_kt, cwd=str(REPO), capture_output=True,
                           text=True)

    if built.returncode != 0:
        print(built.stdout)
        print(built.stderr)
        return 1

    if built.stderr.strip():
        print(built.stderr.strip())

    # 3. Run (same environment rules as the Java suite).
    env = os.environ.copy()
    env["NADATA_HOME"] = str(REPO)

    if sys.platform == "win32":
        build_dir = REPO / "native" / "build" / "win32"
        base_py = Path(sys.base_prefix)
        env["PATH"] = (f"{build_dir};{base_py};{base_py / 'DLLs'};"
                       + env.get("PATH", ""))
    else:
        build_dir = (REPO / "native" / "build" /
                     ("darwin" if sys.platform == "darwin" else "linux"))
        env["PATH"] = f"{build_dir}:" + env.get("PATH", "")

    stdlib = (dist / "lib" / "kotlin-stdlib.jar")
    classpath = os.pathsep.join(
        [str(KT_CLASSES), str(JAVA_CLASSES), str(jna_jar), str(stdlib)])

    run_cmd = [
        java_run.find_tool("java"),
        f"-Djna.library.path={build_dir}",
        "-Djna.nosys=true",
        "-classpath", classpath,
        "nadata.kotlin.ConformanceKt",
    ]

    print("[java]", " ".join(run_cmd))
    run = subprocess.run(run_cmd, cwd=str(REPO), env=env, text=True)

    return run.returncode


if __name__ == "__main__":
    sys.exit(main())
