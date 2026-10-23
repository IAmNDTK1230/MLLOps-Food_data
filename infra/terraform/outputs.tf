output "network_name" {
  value = docker_network.foody_devops.name
}

output "volume_names" {
  value = [
    docker_volume.postgres.name,
    docker_volume.kafka.name,
    docker_volume.prometheus.name,
    docker_volume.grafana.name,
  ]
}
