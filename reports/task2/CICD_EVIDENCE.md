# GitHub Actions CI/CD evidence

**CI VERIFIED.** [Run 39729302994](https://github.com/IAmNDTK1230/MLLOps-Food_data/actions/runs/39729302994) completed with **success** on commit `632dc57ddeb5866b14ede313a887621342e86a5d`. This run verifies that pushed commit, not the current local uncommitted pipeline dependency and report edits.

**CD VERIFIED.** [Publish GHCR image run 39728569052](https://github.com/IAmNDTK1230/MLLOps-Food_data/actions/runs/39728569052) completed with **success** for tag `v1.0.0` at commit `d3a8393358233ae930ab1c4f277b1361ac40ad6c`; its “Publish application image” step succeeded. A read-only `docker buildx imagetools inspect ghcr.io/iamndtk1230/mllops-food_data:v1.0.0` resolved the public OCI manifest, including a `linux/amd64` image, with index digest `sha256:6263bd238e55cfe246986da36c8e6216e13952139ffde5adb6cb3f3f571dc421`. This verifies the existing release image, not the current local pipeline fix. No remote production deployment is configured or claimed.

The Task-2 CI/CD checklist item covers verified CI and GHCR release publication. Local edits still require a future push and CI run at their own commit.
