# CI/CD acceptance — 2026-10-21

## Scope and existing state

The existing `.github/workflows/ci.yml` already ran tests and an API-image build on
`main`; it did not cover `develop`, manual dispatch, a source-only checkout, Compose
validation, or versioned GHCR delivery. The existing workflow was improved in place.
`.github/workflows/release.yml` was added. The accepted local demo, models, Foody
data, indexes, prompts, and Streamlit product behavior were not changed.

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and `develop`, pull requests
targeting `main`, and manual dispatch. Repeated runs for the same ref cancel obsolete
CI runs. The Ubuntu jobs use Python 3.12 and pip caching for `requirements-ci.txt`.

| Job | Checks |
| --- | --- |
| `quality-and-tests` | `python -m ruff check .`; `python -m compileall -q src app scripts`; `python -m pytest -q --junitxml=test_results.xml`; `import app.app, app.api`. JUnit XML is uploaded with `actions/upload-artifact@v4`. |
| `docker-build` | `docker compose config --quiet`; build `Dockerfile.rag` as `food-review-intelligence:ci`. |

The CI dependency list excludes model downloads. The only test needing the real
local demo artifacts skips when those assets are absent. The RAG V3 replay test uses
a small synthetic evidence packet. The real `python -m src.cli demo-preflight`
retains its local-artifact checks and is intentionally not run in GitHub CI.

## CD and image contents

`.github/workflows/release.yml` runs only for pushed `v*` tags or manual dispatch.
It validates a `vMAJOR.MINOR.PATCH` version (with optional prerelease), builds
`Dockerfile.rag`, and publishes to `ghcr.io/<owner>/<repo>` using
`docker/login-action@v3`, `docker/metadata-action@v5`, and
`docker/build-push-action@v6`. Stable tag pushes get version, major.minor, and
`latest` tags; prereleases do not get `latest`. Manual dispatch supplies an explicit
version and does not move `latest`. Release publishing is not canceled once started.

The workflow uses the built-in `GITHUB_TOKEN`, with `contents: read` and
`packages: write`; no PAT or hardcoded credential is required. This is GHCR image
delivery. There is no verified remote production server or automatic deployment.

`Dockerfile.rag` installs `requirements-demo.txt` and copies only application
source, prompts, and configuration. `.dockerignore` excludes `data/`, `models/`,
`state/`, `reports/`, `logs/`, `.env*`, `.git/`, MLflow state, `*.gguf`,
`*.sqlite`, and vector stores. The equivalent private and large local artifact
patterns are covered by `.gitignore`. Runtime model/data/index/state resources
remain mounted or configured outside the image; no local assets were deleted.

## Local verification

Python checks used the project's `.venv` Python 3.12 interpreter (the machine's
unqualified `python` command points to Python 3.9).

| Check | Result |
| --- | --- |
| Full `pytest -q` | **PASS** — 287 passed, 1 warning |
| Source-only checkout `pytest -q` without `data/`, `models/`, `reports/`, or state assets | **PASS** — 286 passed, 1 expected local-demo skip, 1 warning |
| `python -m ruff check .` | **PASS** |
| `python -m compileall -q src app scripts` | **PASS** |
| `import app.app, app.api` | **PASS** |
| `docker compose config --quiet` | **PASS** |
| Workflow YAML parse and trigger inspection | **PASS** for local YAML parsing; no GitHub runner validation yet |
| Local Docker image build | **NOT VERIFIED** — Docker Desktop Linux daemon unavailable |
| GitHub Actions and GHCR publication | **NOT YET VERIFIED** — no GitHub repository exists yet |

The current workspace has no `.git` directory. Therefore `git ls-files` cannot
audit what is tracked, and this report makes no tracked-file claim. After Git
initialization, review the staged file list and staged large files before commit.
No GitHub badge was added because repository ownership and URL are unknown.

## Publish and release commands

Create an empty GitHub repository named `shopee-data-mlops-platform` first, then
run these commands from this project directory. Replace `OWNER` with your GitHub
username or organization. Review `git status --short` and the large-file audit
before committing.
The `Select-String` audit should print no paths; inspect and unstage any match
before running `git commit`.

```powershell
git init
git branch -M main
git add .
git status --short
git diff --cached --name-only
git diff --cached --name-only | Select-String -Pattern '\.gguf$|^data/foody/|^state/|\.sqlite$|vector_store/|^models/'
git commit -m "Add CI and GHCR release workflows"
git remote add origin https://github.com/OWNER/shopee-data-mlops-platform.git
git push -u origin main
```

After the first GitHub CI run passes, create and push the release tag:

```powershell
git tag v1.0.0
git push origin v1.0.0
```

The GitHub repository may require Actions workflow permissions for package
publication; verify the first release run and GHCR package before calling CD live.
