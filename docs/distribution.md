# Distribution — adding nadata to a project in one line

The wrapper files are tiny (a header, a jar, a `.kt` file) — the real
weight is the **engine + embedded CPython bundle** (~17-41 MB per OS).
Every channel below ships that bundle; nothing else is needed at
runtime. Three supported channels:

| Channel | Languages | One-liner |
|---|---|---|
| Install script | C/C++ (and any) | `curl ... \| sh` / `irm ... \| iex` |
| Fat jar | Java, Kotlin, any JVM | drop `nadata-jvm-<version>.jar` on the classpath |
| Staged tarball | everything | manual download from the GitHub release |

## 1. Install scripts (C/C++ and beyond)

```sh
# Linux/macOS
curl -fsSL https://raw.githubusercontent.com/nath54/Nadata/main/scripts/install.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/nath54/Nadata/main/scripts/install.ps1 | iex
```

Options: `--tag v0.1.0`, `--dir third_party/nadata`, `--repo <slug>`,
`--url <url>` (offline/custom source; the PowerShell script uses
`-SourceFile`).

The script detects the platform, downloads the matching
`nadata-native-<target>.tar.gz` from the release, verifies SHA256
against the release `SHA256SUMS`, extracts the staged layout (it
normalizes both the bare layout and CI's `native/<target>/` prefix)
and prints the integration snippet.

### CMake consumption

From `v0.1.0+1` the tarball contains `nadataConfig.cmake`:

```cmake
include(${CMAKE_SOURCE_DIR}/third_party/nadata/nadataConfig.cmake)

add_executable(app main.c)
target_link_libraries(app PRIVATE nadata::nadata)
```

- **POSIX**: rpath entries into the bundle are added automatically —
  nothing else required.
- **Windows (dev)**: run with the bundle on `PATH`
  (`nadata-env.bat` written by the installer), or deploy a
  self-contained copy next to the executable with
  `nadata_deploy_bundle(app)` (production).

Keep the bundle directory intact: the shim resolves `nadata/` (engine)
and `runtime/` (CPython) relative to its own location.

Verified: a 2-line CMake consumer builds against the released
tarball and runs a query (`birthplace=Paris`).

## 2. Fat jar (Java / Kotlin / any JVM)

```sh
python scripts/build_jvm_bundle.py \
    --staged .native-cache/dist/x86_64-pc-windows-msvc \
    --out dist/nadata-jvm-0.1.0.jar          # repeat --staged for a
                                             # multi-platform jar
```

The jar embeds the Java binding classes, JNA (shaded, including its
`jnidispatch.dll`), and one or more complete staged bundles under
`nadata-native/<target>/`. It is the **only classpath entry needed**:

```java
import nadata.bundled.NadataBundled;
import nadata.Nadata;

NadataBundled.install();   // extracts once to ~/.nadata/native/<version>/

try (Nadata.Database db = Nadata.Database.load("./examples/ex0_simple")) {
    ...
}
```

- The loader detects the platform, extracts to
  `~/.nadata/native/<version>/<target>/` (override with
  `-Dnadata.cache=...`), **preloads libpython by absolute path** (so
  no `PATH`/`LD_LIBRARY_PATH`/`DYLD_*` manipulation is needed) and
  loads the shim from the extracted layout.
- Kotlin consumers: additionally drop
  `native/wrappers/kotlin/nadata/NadataExt.kt` into their sources for
  the idiomatic layer (operators, `Long?` optionals, `using {}`).
- Distribution channel for the jar: GitHub release asset for now;
  JitPack/GitHub Packages (`implementation("...")`) is the natural
  next step — the jar is build-system agnostic.

Verified: a consumer with only the jar on the classpath, a clean
environment (no `NADATA_HOME`, no `PATH` injection), extracts and
runs a query (`birthplace=Paris`, `humans=3`), idempotently.

## 3. Raw tarball (everything else)

Download `nadata-native-<target>.tar.gz` from the release, extract,
point your build at it:

- **C**: `-I<dir>` + link `nadata.lib`/`-lnadata` (+ rpath/PATH per
  the config above).
- **C++**: same + `#include "nadata.hpp"` (header-only wrapper).
- **Python hosts** (ctypes): load `<dir>/nadata.dll` etc. — see
  `docs/native.md`.

## Release engineering notes

- Tarballs are produced by the `[dist]` CI jobs (4-OS matrix) and
  attached to the GitHub release next to `SHA256SUMS`.
- `scripts/build_native.py` copies `nadataConfig.cmake` into every
  staged layout; the install scripts tolerate both staged layouts.
- Upgrades are clean: the installers replace the target directory and
  the fat jar caches per version (`~/.nadata/native/<version>/`), so
  old versions keep working.
- Planned follow-ups: JitPack/Gradle coordinates, vcpkg/conan ports,
  `nadata-jvm` on Maven Central.
