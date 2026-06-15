output "server_ip" {
  value = hcloud_server.rag.ipv4_address
}

output "ssh_command" {
  value = "ssh root@${hcloud_server.rag.ipv4_address}"
}

output "api_url" {
  value = "http://${hcloud_server.rag.ipv4_address}:8000"
}

output "grafana_url" {
  value = "http://${hcloud_server.rag.ipv4_address}:3000"
}
