output "deployment_name" {
  description = "Name of the Deployment (or DaemonSet) running the agent. Same value either way — `kubectl get deploy/daemonset -n <ns> <this>`."
  value = (
    var.deploy_as == "deployment"
    ? (length(kubernetes_deployment.agent) > 0 ? kubernetes_deployment.agent[0].metadata[0].name : null)
    : (length(kubernetes_daemonset.agent) > 0 ? kubernetes_daemonset.agent[0].metadata[0].name : null)
  )
}

output "deployment_kind" {
  description = "Whether the agent runs as a Deployment or a DaemonSet (echo of var.deploy_as)."
  value       = var.deploy_as
}

output "namespace" {
  description = "Kubernetes namespace the agent runs in."
  value       = kubernetes_namespace.agent.metadata[0].name
}

output "kubernetes_service_account_name" {
  description = "Name of the K8s SA the pod runs as. Carries the `iam.gke.io/gcp-service-account` annotation that wires Workload Identity."
  value       = kubernetes_service_account.agent.metadata[0].name
}

output "gcp_service_account_email" {
  description = "GCP service account the K8s SA impersonates. Cloud Logging audit entries for Secret Manager + Cost Management calls show this principal."
  value       = google_service_account.agent.email
}

output "workload_identity_member" {
  description = "The IAM principal that has `roles/iam.workloadIdentityUser` on the GCP SA. Useful when debugging `iam.serviceAccounts.signBlob` denials in Cloud Logging."
  value       = "serviceAccount:${local.workload_pool}[${var.namespace}/${local.k8s_sa_name}]"
}

output "secret_agent_api_key_name" {
  description = "Resource name of the Secret Manager secret holding the agent API key."
  value       = google_secret_manager_secret.agent_api_key.name
}

output "secret_tenant_id_name" {
  description = "Resource name of the Secret Manager secret holding the tenant UUID."
  value       = google_secret_manager_secret.tenant_id.name
}

output "verify_registration_hint" {
  description = "Commands to confirm the agent registered with the backend. Mirrors the AWS/Azure/GCE modules' guidance."
  value = join("\n", [
    "kubectl -n ${var.namespace} get pods -l app.kubernetes.io/instance=${var.name}",
    "kubectl -n ${var.namespace} logs -l app.kubernetes.io/instance=${var.name} --tail=200 | grep -i registered",
  ])
}
