$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'iac_down.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Review-2 stop failed.' }
