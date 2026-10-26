#!/usr/bin/env sh
# install.sh — one-line setup for a staged Nadata distribution.
#
#   curl -fsSL https://raw.githubusercontent.com/nath54/Nadata/main/scripts/install.sh | sh
#   # or with options:
#   sh install.sh --tag v0.1.0 --dir third_party/nadata
#
# Detects the platform, downloads the matching release tarball,
# verifies its SHA256 and extracts it into --dir. Then prints the
# integration snippet for your build system.
#
# Options:
#   --tag  <tag>   release tag (default: v0.1.0)
#   --dir  <path>  install directory (default: third_party/nadata)
#   --repo <slug>  GitHub repository (default: nath54/Nadata)
#   --url  <url>   fetch the tarball from this URL instead of a release

# Import Modules
set -eu

REPO="nath54/Nadata"
TAG="v0.1.0"
DIR="third_party/nadata"
URL=""

while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG="$2"; shift 2 ;;
        --dir) DIR="$2"; shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        --url) URL="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# ------------------------------------------------------------------ #
# Platform detection
# ------------------------------------------------------------------ #

case "$(uname -s)" in
    Linux*) os="linux" ;;
    Darwin*) os="darwin" ;;
    *)
        echo "error: use scripts/install.ps1 on Windows" >&2
        exit 1
        ;;
esac

case "$(uname -m)" in
    x86_64|amd64) arch="x86_64" ;;
    aarch64|arm64) arch="aarch64" ;;
    *) echo "error: unsupported arch $(uname -m)" >&2; exit 1 ;;
esac

TARGET="${arch}-unknown-${os}-gnu"
if [ "$os" = "darwin" ]; then
    TARGET="${arch}-apple-darwin"
fi

TARBALL="nadata-native-${TARGET}.tar.gz"
if [ -z "$URL" ]; then
    URL="https://github.com/${REPO}/releases/download/${TAG}/${TARBALL}"
fi

echo "[nadata] target ${TARGET}"
echo "[nadata] fetching ${URL}"

# ------------------------------------------------------------------ #
# Download (curl or wget) + checksum + extract
# ------------------------------------------------------------------ #

TMP="$(mktemp -d)"
mkdir -p "$TMP/unpack"
trap 'rm -rf "$TMP"' EXIT

if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$TMP/$TARBALL" "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget -qO "$TMP/$TARBALL" "$URL"
else
    echo "error: need curl or wget" >&2
    exit 1
fi

SUMS_URL="https://github.com/${REPO}/releases/download/${TAG}/SHA256SUMS"

if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$TMP/SHA256SUMS" "$SUMS_URL" || true
elif command -v wget >/dev/null 2>&1; then
    wget -qO "$TMP/SHA256SUMS" "$SUMS_URL" || true
fi

if [ -f "$TMP/SHA256SUMS" ]; then
    EXPECTED="$(grep " $TARBALL\$" "$TMP/SHA256SUMS" | cut -d' ' -f1)"

    if [ -n "$EXPECTED" ]; then
        if command -v sha256sum >/dev/null 2>&1; then
            ACTUAL="$(sha256sum "$TMP/$TARBALL" | cut -d' ' -f1)"
        else
            ACTUAL="$(shasum -a 256 "$TMP/$TARBALL" | cut -d' ' -f1)"
        fi

        if [ "$ACTUAL" != "$EXPECTED" ]; then
            echo "error: checksum mismatch for $TARBALL" >&2
            exit 1
        fi

        echo "[nadata] checksum ok"
    fi
fi

mkdir -p "$DIR"
tar -xzf "$TMP/$TARBALL" -C "$TMP/unpack"

# Normalise both layouts: a bare staged root, or CI's native/<target>/
# prefix. The staged root is the directory containing nadata.h.
ROOT="$TMP/unpack"

if [ ! -f "$ROOT/nadata.h" ]; then
    FOUND="$(find "$TMP/unpack" -name nadata.h -type f | head -1)"

    if [ -z "$FOUND" ]; then
        echo "error: no nadata.h in $TARBALL" >&2
        exit 1
    fi

    ROOT="$(dirname "$FOUND")"
fi

# Clear any previous install so stale files cannot shadow the new one.
rm -rf "$DIR"
mkdir -p "$DIR"
cp -R "$ROOT"/. "$DIR/"

echo "[nadata] installed into $DIR"

# ------------------------------------------------------------------ #
# Integration snippets
# ------------------------------------------------------------------ #

ABS_DIR="$(cd "$DIR" && pwd)"

cat <<EOF

[nadata] done. Integrate with your build:

CMake (C/C++):
  include(${ABS_DIR}/nadataConfig.cmake)
  target_link_libraries(your_app PRIVATE nadata::nadata)

  # POSIX: nothing else (rpath is set up for you).
  # Windows (MSVC): add the bundle to PATH when running, e.g.
  #   set PATH=${ABS_DIR};%PATH%
  # or deploy a self-contained copy: nadata_deploy_bundle(your_app)

Java/Kotlin (JVM):
  Put nadata-jvm-<version>.jar on the classpath and call
  nadata.bundled.NadataBundled.install() once before first use
  (self-extracting; see docs/distribution.md).

Header-only quick test:
  ${ABS_DIR}/nadata.h   (C ABI)
  ${ABS_DIR}/nadata.hpp (C++ wrapper)
EOF
