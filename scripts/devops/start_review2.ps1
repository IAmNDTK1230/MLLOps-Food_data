param([switch]$ProduceEvents)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'iac_up.ps1') -ProduceEvents:$ProduceEvents
if ($LASTEXITCODE -ne 0) { throw 'Review-2 startup failed.' }
