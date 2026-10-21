param(
    [int]$Port = 8501,
    [switch]$CpuOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Demo Python environment is missing: .venv\Scripts\python.exe'
}

Write-Host 'Checking frozen demo artifacts...'
& $python -m src.cli demo-preflight
if ($LASTEXITCODE -ne 0) {
    throw 'Demo preflight failed. Restore the listed artifact before starting.'
}

$launchJson = & $python -m scripts.demo_launch_info
if ($LASTEXITCODE -ne 0) { throw 'Cannot read frozen Qwen launch configuration.' }
$launch = $launchJson | ConvertFrom-Json
$url = "http://127.0.0.1:$($launch.port)"
$ownedServer = $null

function Test-DemoServer {
    try {
        $null = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

try {
    if (Test-DemoServer) {
        $properties = Invoke-RestMethod -Uri "$url/props" -TimeoutSec 3
        if (($properties | ConvertTo-Json -Depth 6) -notlike "*$($launch.model_file)*") {
            throw "Port $($launch.port) is serving another model. Stop it before the demo."
        }
        Write-Host 'Using the existing verified Qwen 3B server.'
    } else {
        $layers = if ($CpuOnly) { 0 } else { [int]$launch.gpu_layers }
        New-Item -ItemType Directory -Path 'logs' -Force | Out-Null
        $arguments = @('--model', ('"' + $launch.model + '"'), '--host', '127.0.0.1',
                       '--port', [string]$launch.port, '--n-gpu-layers', [string]$layers,
                       '--threads', [string]$launch.threads,
                       '--threads-batch', [string]$launch.threads,
                       '--ctx-size', [string]$launch.context_size,
                       '--parallel', '1', '--batch-size', '512', '--ubatch-size', '256')
        $ownedServer = Start-Process -FilePath $launch.executable -ArgumentList $arguments `
            -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput 'logs\demo_llama_stdout.log' `
            -RedirectStandardError 'logs\demo_llama_stderr.log'
        $ready = $false
        for ($attempt = 0; $attempt -lt 60; $attempt++) {
            if (Test-DemoServer) { $ready = $true; break }
            if ($ownedServer.HasExited) { break }
            Start-Sleep -Seconds 1
        }
        if (-not $ready) {
            throw 'Qwen server did not become healthy. Inspect logs\demo_llama_stderr.log.'
        }
        Write-Host "Started Qwen 3B server on $url."
    }
    $env:QWEN_BASE_URL = $url
    Write-Host "Starting Streamlit at http://localhost:$Port"
    & $python -m streamlit run app/app.py --server.port $Port
    exit $LASTEXITCODE
} finally {
    if ($ownedServer -and -not $ownedServer.HasExited) {
        Stop-Process -Id $ownedServer.Id
        Write-Host 'Stopped the demo-owned Qwen server.'
    }
}
