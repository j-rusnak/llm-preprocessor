# LLM Preprocessor — Benchmark Runner
#
# Builds the benchmark, runs it from the project root (so config.json
# and models/ are accessible), and generates visualization PNGs.
#
# Prerequisites:
#   - cmake --build build  (project must be built first)
#   - pip install matplotlib numpy
#
# Usage (from project root):
#   .\benchmarks\run_benchmarks.ps1

$ErrorActionPreference = "Stop"

# Resolve paths
$scriptDir   = $PSScriptRoot
$projectRoot = Split-Path $scriptDir -Parent
$buildDir    = Join-Path $projectRoot "build"
$resultsDir  = Join-Path $scriptDir "results"

# Create results directory
New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null

# Check executable exists
$benchExe = Join-Path $buildDir "benchmark_runner.exe"
if (-not (Test-Path $benchExe)) {
    Write-Host "Error: benchmark_runner.exe not found. Build the project first:" -ForegroundColor Red
    Write-Host "  cmake --build build" -ForegroundColor Yellow
    exit 1
}

# Run benchmark from project root so it can find config.json and models/
Write-Host "`n=== Running C++ Benchmark ===" -ForegroundColor Cyan
Push-Location $projectRoot
try {
    & $benchExe > "$resultsDir\benchmark_data.json"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Benchmark failed!" -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}
Write-Host "Benchmark data saved to: $resultsDir\benchmark_data.json" -ForegroundColor Green

# Generate visualizations
Write-Host "`n=== Generating Visualizations ===" -ForegroundColor Cyan
$vizScript = Join-Path $scriptDir "visualize.py"
python $vizScript "$resultsDir\benchmark_data.json" $resultsDir

if ($LASTEXITCODE -ne 0) {
    Write-Host "Visualization failed! Make sure matplotlib and numpy are installed:" -ForegroundColor Red
    Write-Host "  pip install matplotlib numpy" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host "Results directory: $resultsDir"
Write-Host "Open the PNG files for presentation-ready charts.`n"
