output "instance_name" {
  description = "Name of the GCE instance running the agent. Feed to `gcloud compute instances describe / ssh / tail-serial-port-output`."
  value       = google_compute_instance.agent.name
}

output "instance_self_link" {
  description = "Fully-qualified self-link of the GCE instance. Stable identifier suitable for cross-stack references."
  value       = google_compute_instance.agent.self_link
}

output "instance_zone" {
  description = "Zone the VM was launched in (echo of var.zone after validation)."
  value       = google_compute_instance.agent.zone
}

output "service_account_email" {
  description = "Email of the service account attached to the VM. The audit trail in Cloud Logging will show this principal making Secret Manager + Cost Management calls."
  value       = google_service_account.agent.email
}

output "secret_agent_api_key_name" {
  description = "Resource name of the Secret Manager secret holding the agent API key. The VM's SA has accessor rights scoped to JUST this secret."
  value       = google_secret_manager_secret.agent_api_key.name
}

output "secret_tenant_id_name" {
  description = "Resource name of the Secret Manager secret holding the tenant UUID."
  value       = google_secret_manager_secret.tenant_id.name
}

output "install_log_path" {
  description = "Path on the VM where the startup-script writes its log."
  value       = "/var/log/tensorcost-agent-install.log"
}

output "agent_service_name" {
  description = "systemd unit name the agent runs under. Tail with `journalctl -u unified-gpu-agent -f` (over `gcloud compute ssh`)."
  value       = "unified-gpu-agent.service"
}

output "verify_registration_hint" {
  description = "How to confirm the agent registered with the backend. Mirrors the AWS/Azure modules' guidance."
  value       = "gcloud compute ssh ${google_compute_instance.agent.name} --zone=${google_compute_instance.agent.zone} -- 'sudo journalctl -u unified-gpu-agent -n 50 | grep -i registered'"
}
