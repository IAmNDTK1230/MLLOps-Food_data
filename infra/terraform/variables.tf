variable "docker_host" {
  type        = string
  description = "Docker Engine endpoint; defaults to Docker Desktop on Windows."
  default     = "npipe:////./pipe/dockerDesktopLinuxEngine"
}
