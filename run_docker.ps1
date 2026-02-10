# ============================================================================
# Unsloth GPU Benchmark — Docker Runner (Windows PowerShell)
# ============================================================================
# Usage:
#   .\run_docker.ps1                          # Interactive: Jupyter Lab
#   .\run_docker.ps1 -Mode run -Profile large # Headless: runs benchmark
#
# Profiles: small | medium | large | xlarge | stress | all
# ============================================================================

param(
    [string]$Mode = "interactive",
    [string]$Profile = "large"
)

$ContainerName = "unsloth-benchmark"
$WorkDir = Join-Path (Get-Location) "work"

# Create work directory
if (-not (Test-Path $WorkDir)) {
    New-Item -ItemType Directory -Path $WorkDir | Out-Null
}

# Copy benchmark script to work directory
Copy-Item -Path "benchmark.py" -Destination (Join-Path $WorkDir "benchmark.py") -Force

Write-Host "============================================================"
Write-Host "  Unsloth GPU Benchmark - Docker Setup"
Write-Host "============================================================"
Write-Host "  Mode:      $Mode"
Write-Host "  Profile:   $Profile"
Write-Host "  Work Dir:  $WorkDir"
Write-Host "============================================================"

# Stop existing container
docker rm -f $ContainerName 2>$null

if ($Mode -eq "run") {
    Write-Host ""
    Write-Host "Running benchmark headless (profile: $Profile)..."
    Write-Host ""
    docker run --rm `
        --name $ContainerName `
        --entrypoint bash `
        -v "${WorkDir}:/workspace/work" `
        --gpus all `
        unsloth/unsloth `
        -c "cd /workspace/work && python benchmark.py --profile $Profile"
}
else {
    Write-Host ""
    Write-Host "Starting Unsloth container with Jupyter Lab..."
    Write-Host ""
    docker run -d `
        --name $ContainerName `
        -e JUPYTER_PASSWORD="benchmark" `
        -p 8888:8888 -p 2222:22 `
        -v "${WorkDir}:/workspace/work" `
        --gpus all `
        unsloth/unsloth

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "  Container started!"
    Write-Host "  Jupyter Lab:  http://localhost:8888  (password: benchmark)"
    Write-Host ""
    Write-Host "  To run the benchmark inside the container:"
    Write-Host "    docker exec -it $ContainerName bash"
    Write-Host "    cd /workspace/work && python benchmark.py --profile $Profile"
    Write-Host ""
    Write-Host "  Or run headless:"
    Write-Host "    .\run_docker.ps1 -Mode run -Profile $Profile"
    Write-Host "============================================================"
}
