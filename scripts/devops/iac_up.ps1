param([switch]$ProduceEvents)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $root
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker CLI is required.' }
docker info *> $null
if ($LASTEXITCODE -ne 0) { throw 'Docker daemon is unavailable.' }
$terraformCmd = Get-Command terraform -ErrorAction SilentlyContinue
if ($terraformCmd) {
    $terraformExe = $terraformCmd.Source
} else {
    $cachedTerraform = Join-Path $root 'tmp/experiments/terraform-bin/terraform.exe'
    if (-not (Test-Path -LiteralPath $cachedTerraform)) {
        throw 'Terraform is required to provision the Review-2 network and volumes.'
    }
    $terraformExe = $cachedTerraform
}
& $terraformExe -chdir=infra/terraform init -input=false
if ($LASTEXITCODE -ne 0) { throw 'Terraform init failed.' }
& $terraformExe -chdir=infra/terraform apply -input=false -auto-approve
if ($LASTEXITCODE -ne 0) { throw 'Terraform apply failed.' }
$compose = @('-f', 'docker-compose.yml', '-f', 'docker-compose.devops.yml')
$baseImage = if ($env:FOODY_BASE_IMAGE) { $env:FOODY_BASE_IMAGE } else { "$(Split-Path $root -Leaf)-rag-backend:latest" }
docker image inspect $baseImage *> $null
if ($LASTEXITCODE -ne 0) {
    docker compose -f docker-compose.yml build rag-backend
    if ($LASTEXITCODE -ne 0) { throw 'Base RAG image build failed.' }
}
$env:FOODY_BASE_IMAGE = $baseImage
$env:GIT_SHA = (git rev-parse HEAD).Trim()
docker compose @compose build rag-backend
if ($LASTEXITCODE -ne 0) { throw 'Small DevOps image build failed.' }
docker compose @compose up -d --wait --wait-timeout 240 postgres kafka qwen rag-backend streamlit prometheus grafana
if ($LASTEXITCODE -ne 0) { throw 'Compose startup failed.' }
$hasTable = (docker compose @compose exec -T postgres psql -U foody -d foody -Atc `
    "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='fact_review')").Trim()
$needsLoad = $hasTable -ne 't'
if (-not $needsLoad) {
    $count = [int](docker compose @compose exec -T postgres psql -U foody -d foody -Atc `
        'SELECT COUNT(*) FROM fact_review').Trim()
    $version = (docker compose @compose exec -T postgres psql -U foody -d foody -Atc `
        'SELECT MAX(version) FROM schema_migrations').Trim()
    $needsLoad = $count -eq 0 -or $version -ne '002'
}
if ($needsLoad) {
    docker compose @compose exec -T rag-backend python -m src.cli warehouse-load
    if ($LASTEXITCODE -ne 0) { throw 'Warehouse load failed.' }
} else {
    Write-Host "Warehouse already loaded ($count reviews; migration $version)."
}
docker compose @compose exec -T kafka /opt/kafka/bin/kafka-topics.sh `
    --bootstrap-server localhost:19092 --create --if-not-exists `
    --topic foody.review.events --partitions 1 --replication-factor 1 `
    --config retention.ms=21600000 --config retention.bytes=134217728 `
    --config segment.bytes=16777216
if ($LASTEXITCODE -ne 0) { throw 'Kafka topic creation failed.' }
docker compose @compose up -d --wait --wait-timeout 120 kafka-consumer
if ($LASTEXITCODE -ne 0) { throw 'Kafka consumer startup failed.' }
if ($ProduceEvents) {
    docker compose @compose exec -T rag-backend python -m src.cli kafka-produce --limit 20
    if ($LASTEXITCODE -ne 0) { throw 'Kafka demo replay failed.' }
}
docker compose @compose ps
Write-Host 'Streamlit http://localhost:8501 | Grafana http://localhost:3000 | Prometheus http://localhost:9090'
