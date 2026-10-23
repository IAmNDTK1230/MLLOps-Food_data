$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $root
$compose = @('-f', 'docker-compose.yml', '-f', 'docker-compose.devops.yml')
docker compose @compose stop kafka-consumer grafana prometheus kafka postgres
if ($LASTEXITCODE -ne 0) { throw 'Could not stop Review-2 services.' }
Write-Host 'Review-2 services stopped. Docker network, named volumes, and normal demo remain.'
