provider "docker" {
  host = var.docker_host
}

resource "docker_network" "foody_devops" {
  name   = "foody_devops_network"
  driver = "bridge"
}

resource "docker_volume" "postgres" {
  name = "foody_postgres_data"
}

resource "docker_volume" "kafka" {
  name = "foody_kafka_data"
}

resource "docker_volume" "prometheus" {
  name = "foody_prometheus_data"
}

resource "docker_volume" "grafana" {
  name = "foody_grafana_data"
}
