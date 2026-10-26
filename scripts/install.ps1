# install.ps1 — one-line setup for a staged Nadata distribution (Windows).
#
#   irm https://raw.githubusercontent.com/nath54/Nadata/main/scripts/install.ps1 | iex
#   # or with options:
#   .\scripts\install.ps1 -Tag v0.1.0 -Dir third_party\nadata
#   # offline / custom source:
#   .\scripts\install.ps1 -SourceFile nadata-native-x86_64-pc-windows-msvc.tar.gz
#
# Downloads the matching release tarball, verifies its SHA256 and
# extracts it into -Dir, then prints the integration snippet.

# Import Modules
param(
    [string]$Tag = "v0.1.0",
    [string]$Dir = "third_party\nadata",
    [string]$Repo = "nath54/Nadata",
    [string]$SourceFile = ""
)

$ErrorActionPreference = "Stop"

# ------------------------------------------------------------------ #
# Platform detection
# ------------------------------------------------------------------ #

$arch = switch ($env:PROCESSOR_ARCHITECTURE) {
    "AMD64" { "x86_64" }
    "ARM64" { "aarch64" }
    default { throw "unsupported architecture: $env:PROCESSOR_ARCHITECTURE" }
}

$Target = "$arch-pc-windows-msvc"
$Tarball = "nadata-native-$Target.tar.gz"

# ------------------------------------------------------------------ #
# Download (or copy) + checksum + extract
# ------------------------------------------------------------------ #

$tmp = New-Item -ItemType Directory -Path ([IO.Path]::GetTempPath() + [IO.Path]::GetRandomFileName())

try {
    if ($SourceFile -ne "") {
        Copy-Item $SourceFile "$tmp/$Tarball"
    }
    else {
        $Url = "https://github.com/$Repo/releases/download/$Tag/$Tarball"
        Write-Host "[nadata] target $Target"
        Write-Host "[nadata] fetching $Url"
        Invoke-WebRequest -Uri $Url -OutFile "$tmp/$Tarball" -UseBasicParsing

        try {
            $Sums = Invoke-WebRequest `
                -Uri "https://github.com/$Repo/releases/download/$Tag/SHA256SUMS" `
                -UseBasicParsing
            $Expected = ($Sums.Content -split "`n" |
                Where-Object { $_ -match [regex]::Escape($Tarball) } |
                Select-Object -First 1) -split ' ' | Select-Object -First 1

            if ($Expected) {
                $Actual = (Get-FileHash -Algorithm SHA256 `
                    "$tmp/$Tarball").Hash.ToLower()
                if ($Actual -ne $Expected) {
                    throw "checksum mismatch for $Tarball"
                }
                Write-Host "[nadata] checksum ok"
            }
        }
        catch [System.Exception] {
            Write-Host "[nadata] warning: could not verify SHA256SUMS"
        }
    }

    New-Item -ItemType Directory -Force -Path $Dir | Out-Null

    # Windows 10+ ships bsdtar; use the absolute path so a GNU tar
    # from git-bash/msys on PATH cannot hijack the call.
    $TarExe = "C:\Windows\System32\tar.exe"
    if (-not (Test-Path $TarExe)) { $TarExe = "tar" }
    $unpack = "$tmp/unpack"
    New-Item -ItemType Directory -Force -Path $unpack | Out-Null
    & $TarExe -xzf "$tmp/$Tarball" -C $unpack

    # Normalise both layouts: a bare staged root, or CI's
    # native/<target>/ prefix. The staged root contains nadata.h.
    $Root = $unpack
    if (-not (Test-Path "$unpack/nadata.h")) {
        $Header = Get-ChildItem -Path $unpack -Recurse `
            -Filter nadata.h -File | Select-Object -First 1

        if (-not $Header) {
            throw "no nadata.h in $Tarball"
        }

        $Root = $Header.DirectoryName
    }

    if (Test-Path $Dir) { Remove-Item -Recurse -Force $Dir }
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    Copy-Item -Path "$Root/*" -Destination $Dir -Recurse -Force
}
finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

Write-Host "[nadata] installed into $Dir"

# ------------------------------------------------------------------ #
# Integration helper: a batch file that puts the bundle on PATH
# ------------------------------------------------------------------ #

$AbsDir = (Resolve-Path $Dir).Path
$envBat = Join-Path $Dir "nadata-env.bat"
@"
@echo off
rem Adds the nadata bundle to PATH for this session (run: nadata-env.bat)
set PATH=$AbsDir;%PATH%
"@ | Set-Content -Path $envBat -Encoding ASCII

Write-Host @"

[nadata] done. Integrate with your build:

CMake (C/C++):
  include($AbsDir/nadataConfig.cmake)
  target_link_libraries(your_app PRIVATE nadata::nadata)

  # dev runs: nadata-env.bat  (puts the bundle on PATH)
  # or a self-contained deploy: nadata_deploy_bundle(your_app)

Java/Kotlin (JVM):
  Put nadata-jvm-<version>.jar on the classpath and call
  nadata.bundled.NadataBundled.install() once before first use
  (self-extracting; see docs/distribution.md).

Header-only quick test:
  $AbsDir/nadata.h   (C ABI)
  $AbsDir/nadata.hpp (C++ wrapper)
"@
