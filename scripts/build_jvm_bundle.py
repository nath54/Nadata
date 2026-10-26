"""
build_jvm_bundle.py — packages the JVM binding + staged bundles into
one self-extracting jar (nadata-jvm-<version>.jar).

Contents:
    nadata/*.class                       Java binding (JNA)
    nadata/bundled/NadataBundled.class   self-extracting loader
    nadata-native/version.txt            version marker
    nadata-native/<target>/...           complete staged bundles
                                         (shim + nadata/ + runtime/)

Consumers put the jar on the classpath and call
nadata.bundled.NadataBundled.install() once (see the class doc).
Multiple --staged inputs produce a multi-platform jar.

Usage:
    python scripts/build_jvm_bundle.py \
        --staged .native-cache/dist/x86_64-pc-windows-msvc \
        [--staged .native-cache/dist/aarch64-apple-darwin ...] \
        [--out dist/nadata-jvm-0.1.1.jar] [--version 0.1.1]

The Java binding + loader are compiled with the host JDK (JAVA_HOME or
PATH); no Gradle/Maven needed.
"""

# Import Modules
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JAVA_SRC = REPO / "native" / "wrappers" / "java" / "nadata"
LOADER_SRC = (REPO / "native" / "wrappers" / "java" / "bundled"
              / "nadata" / "bundled" / "NadataBundled.java")

JNA_VERSION = "5.14.0"
JNA_JAR = REPO / ".native-cache" / "jna" / f"jna-{JNA_VERSION}.jar"


def find_javac() -> str:
    """Locates javac: JAVA_HOME first, then PATH."""

    home = os.environ.get("JAVA_HOME")

    if home:
        candidate = Path(home) / "bin" / ("javac.exe" if os.name == "nt"
                                          else "javac")

        if candidate.exists():
            return str(candidate)

    return "javac"


def main() -> int:
    """Builds the fat jar; returns the process exit code."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="append", required=True,
                        help="staged dist dir (repeatable)")
    parser.add_argument("--out", default=None, help="output jar path")
    parser.add_argument("--version", default="0.1.1")
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path(
        f"dist/nadata-jvm-{args.version}.jar")
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Compile the binding + loader.
    classes = Path(tempfile.mkdtemp(prefix="nadata-jvm-"))

    try:
        compile_cmd = [
            find_javac(), "-encoding", "UTF-8",
            "-classpath", str(JNA_JAR), "-d", str(classes),
            str(JAVA_SRC / "Nadata.java"),
            str(JAVA_SRC / "NadataLibrary.java"),
            str(LOADER_SRC),
        ]

        print("[javac]", " ".join(compile_cmd))
        built = subprocess.run(compile_cmd, cwd=str(REPO),
                               capture_output=True, text=True)

        if built.returncode != 0:
            print(built.stdout)
            print(built.stderr)
            return 1

        # 2. Write the jar: classes + JNA (shaded) + version + bundles.
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as jar:
            for class_file in sorted(classes.rglob("*.class")):
                arcname = class_file.relative_to(classes).as_posix()
                jar.write(class_file, arcname)

            # Embed JNA so the jar is the only classpath entry needed
            # (classes + its bundled jnidispatch.dll resources; drop the
            # signature files so the shaded entries stay valid).
            with zipfile.ZipFile(JNA_JAR) as jna:
                for info in jna.infolist():
                    if (info.filename.startswith("META-INF")
                            or info.filename.endswith("/")):
                        continue

                    jar.writestr(info.filename,
                                 jna.read(info.filename))

            jar.writestr("nadata-native/version.txt", args.version)

            for staged in args.staged:
                staged_path = Path(staged).resolve()
                target = staged_path.name
                print(f"[jar] nadata-native/{target}/ <- {staged_path}")

                for file_path in sorted(staged_path.rglob("*")):
                    if not file_path.is_file():
                        continue

                    # Follow symlinks (staged Linux libs are symlinks).
                    real = file_path.resolve()
                    arcname = (f"nadata-native/{target}/"
                               + file_path.relative_to(staged_path)
                               .as_posix())

                    jar.write(real, arcname)
    finally:
        shutil.rmtree(classes, ignore_errors=True)

    size_mb = out.stat().st_size / (1 << 20)
    print(f"[jar] wrote {out} ({size_mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
