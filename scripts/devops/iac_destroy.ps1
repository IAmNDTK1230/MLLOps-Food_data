param([switch]$ConfirmDestroy)
if (-not $ConfirmDestroy) {
    throw 'Destructive operation. Rerun explicitly with -ConfirmDestroy after stopping Review-2 containers.'
}
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $root
terraform -chdir=infra/terraform destroy -input=false
if ($LASTEXITCODE -ne 0) { throw 'Terraform destroy failed or was cancelled.' }
