# One-command setup on Windows: creates .venv, installs Group Pulse with dev extras, runs the tests.
# Usage:  ./scripts/setup.ps1            (add -Video to also install OpenCV for video checks)
param([switch]$Video)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv")) { python -m venv .venv }
$py = Join-Path $root ".venv\Scripts\python.exe"

& $py -m pip install -U pip
if ($Video) { & $py -m pip install -e ".[dev,video]" } else { & $py -m pip install -e ".[dev]" }
& $py -m pytest -q
