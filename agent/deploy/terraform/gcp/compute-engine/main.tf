/**
 * compute-engine — provision a GCE VM running the unified-gpu-agent.
 *
 * Sibling of the AWS `ec2-agent` and Azure `agent-extension` modules but
 * for Google Cloud. The model is "greenfield single VM" — Terraform
 * creates the instance, the agent's GCP service account, the Secret
 * Manager secrets it reads, and the IAM bindings that let the VM
 * impersonate the SA via Workload Identity Federation (the VM's
 * attached SA — no JSON keys ever leave GCP).
 *
 * Why no service-account JSON key?
 * --------------------------------
 * The §8.3 audit explicitly calls out the Helm chart's GCP path
 * encouraging `gcloud iam service-accounts keys create` + mounting the
 * resulting JSON. Long-lived keys are GCP's #1 credential-leak vector
 * (commit history, container layers, broken Vault rotation). This
 * module instead attaches the SA directly to the VM — `gcloud auth
 * application-default print-access-token` then returns short-lived
 * IAM tokens minted by the metadata server. No key file, ever.
 *
 * Sibling sub-module:
 *   gke-workload/  — Kubernetes Deployment for tenants whose GPUs are
 *                    on an existing GKE cluster. Uses Workload Identity
 *                    on K8s SA → GCP SA binding.
 */

// ─── 1. Service account the VM runs as ────────────────────────────────────

resource "google_service_account" "agent" {
  project      = var.project_id
  account_id   = "${var.name}-sa"
  display_name = "TensorCost unified-gpu-agent (${var.name})"
  description  = "Service account attached to the agent VM. Reads its API key + tenant ID from Secret Manager via the metadata server — no JSON keys."
}

// ─── 2. Project-level IAM — minimum-viable runtime roles ─────────────────
// The agent needs:
//   - roles/monitoring.viewer  — instance metadata + per-VM CPU/mem
//                                signals from Cloud Monitoring
//   - roles/billing.viewer     — only when billing_account_id is provided
//                                (granted on the billing account, not
//                                the project)
//   - extra_iam_roles[]        — caller-supplied extras (BigQuery cost
//                                export, Compute viewer for dashboard
//                                enumeration, etc.)
//
// Roles deliberately NOT granted by default:
//   - roles/cloudkms.admin / roles/iam.serviceAccountAdmin /
//     roles/owner / roles/editor — far too broad. The agent never
//     mutates infra.
//   - roles/compute.viewer at project scope — comes via extra_iam_roles[]
//     when the customer wants per-instance enumeration on the dashboard.
//     Not on by default because some tenants use Cloud Asset Inventory
//     instead.

resource "google_project_iam_member" "monitoring_viewer" {
  project = var.project_id
  role    = "roles/monitoring.viewer"
  member  = "serviceAccount:${google_service_account.agent.email}"
}

resource "google_project_iam_member" "extras" {
  for_each = toset(var.extra_iam_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.agent.email}"
}

resource "google_billing_account_iam_member" "billing_viewer" {
  count = var.billing_account_id != "" ? 1 : 0

  billing_account_id = var.billing_account_id
  role               = "roles/billing.viewer"
  member             = "serviceAccount:${google_service_account.agent.email}"
}

// ─── 3. Secrets in Secret Manager (encrypted, scoped IAM) ────────────────
// One secret per value so the SA's accessor binding can stay
// resource-scoped — leaking the SA's token does not unlock unrelated
// project secrets.

resource "google_secret_manager_secret" "agent_api_key" {
  project   = var.project_id
  secret_id = "${var.name}-agent-api-key"

  labels = local.common_labels

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "agent_api_key" {
  secret      = google_secret_manager_secret.agent_api_key.id
  secret_data = var.agent_api_key
}

resource "google_secret_manager_secret_iam_member" "agent_api_key_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.agent_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent.email}"
}

resource "google_secret_manager_secret" "tenant_id" {
  project   = var.project_id
  secret_id = "${var.name}-tenant-id"

  labels = local.common_labels

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "tenant_id" {
  secret      = google_secret_manager_secret.tenant_id.id
  secret_data = var.tenant_id
}

resource "google_secret_manager_secret_iam_member" "tenant_id_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.tenant_id.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent.email}"
}

resource "google_secret_manager_secret" "agent_key_id" {
  count     = var.grpc_host != "" ? 1 : 0
  project   = var.project_id
  secret_id = "${var.name}-agent-key-id"

  labels = local.common_labels

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "agent_key_id" {
  count       = var.grpc_host != "" ? 1 : 0
  secret      = google_secret_manager_secret.agent_key_id[0].id
  secret_data = var.agent_key_id
}

resource "google_secret_manager_secret_iam_member" "agent_key_id_accessor" {
  count     = var.grpc_host != "" ? 1 : 0
  project   = var.project_id
  secret_id = google_secret_manager_secret.agent_key_id[0].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent.email}"
}

resource "google_secret_manager_secret" "agent_hmac_pepper" {
  count     = var.grpc_host != "" ? 1 : 0
  project   = var.project_id
  secret_id = "${var.name}-agent-hmac-pepper"

  labels = local.common_labels

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "agent_hmac_pepper" {
  count       = var.grpc_host != "" ? 1 : 0
  secret      = google_secret_manager_secret.agent_hmac_pepper[0].id
  secret_data = var.agent_hmac_pepper
}

resource "google_secret_manager_secret_iam_member" "agent_hmac_pepper_accessor" {
  count     = var.grpc_host != "" ? 1 : 0
  project   = var.project_id
  secret_id = google_secret_manager_secret.agent_hmac_pepper[0].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent.email}"
}

// ─── 4. Startup script — same shape as the AWS / Azure install scripts ───

locals {
  # Mirror the AWS module's digest-pinning pattern (commit ead77a7).
  # If the caller supplied a sha256 digest, rewrite `repo:tag` →
  # `repo@sha256:...` so docker pulls a content-addressed reference.
  agent_image_no_tag = (
    var.agent_image_digest != "" && length(regexall(":[^/]+$", var.agent_image)) > 0
    ? replace(var.agent_image, "/:[^/]+$/", "")
    : var.agent_image
  )
  agent_image_ref = (
    var.agent_image_digest != ""
    ? "${local.agent_image_no_tag}@${var.agent_image_digest}"
    : var.agent_image
  )

  startup_script = templatefile("${path.module}/startup-script.sh.tftpl", {
    backend_api_url               = var.backend_api_url
    grpc_host                     = var.grpc_host
    grpc_port                     = var.grpc_port
    metrics_push_interval         = var.metrics_push_interval
    agent_image                   = local.agent_image_ref
    agent_image_digest_pinned     = var.agent_image_digest != ""
    use_private_artifact_registry = var.use_private_artifact_registry
    artifact_registry_host        = var.artifact_registry_host
    project_id                    = var.project_id
    secret_agent_api_key          = google_secret_manager_secret.agent_api_key.secret_id
    secret_tenant_id              = google_secret_manager_secret.tenant_id.secret_id
    secret_agent_key_id           = var.grpc_host != "" ? google_secret_manager_secret.agent_key_id[0].secret_id : ""
    secret_agent_hmac_pepper      = var.grpc_host != "" ? google_secret_manager_secret.agent_hmac_pepper[0].secret_id : ""
    agent_hostname                = local.resolved_agent_hostname
    has_gpu                       = var.gpu_type != ""
    grpc_enabled                  = var.grpc_host != ""
  })
}

// ─── 5. The VM itself ─────────────────────────────────────────────────────

resource "google_compute_instance" "agent" {
  project      = var.project_id
  name         = var.name
  zone         = var.zone
  machine_type = var.machine_type

  labels = local.common_labels

  boot_disk {
    initialize_params {
      image  = var.boot_image
      size   = var.boot_disk_size_gb
      type   = "pd-balanced"
      labels = local.common_labels
    }
  }

  network_interface {
    network    = var.network
    subnetwork = var.subnetwork != "" ? var.subnetwork : null

    # Ephemeral public IP — needed for the default-network case to reach
    # docker.io / Secret Manager. Air-gapped tenants should drop this
    # (set via a fork) and use Private Google Access on the subnet.
    access_config {}
  }

  // Workload Identity for VMs = attach the service account to the VM
  // and grant cloud-platform scope. The metadata server then mints
  // short-lived OAuth tokens for `gcloud auth application-default`
  // and the agent SDK clients automatically. NO JSON KEY EVER ISSUED.
  service_account {
    email  = google_service_account.agent.email
    scopes = ["cloud-platform"]
  }

  // Accelerators require host-maintenance=TERMINATE (GCE constraint).
  dynamic "guest_accelerator" {
    for_each = var.gpu_type != "" ? [1] : []
    content {
      type  = var.gpu_type
      count = var.gpu_count
    }
  }

  scheduling {
    on_host_maintenance = var.gpu_type != "" ? "TERMINATE" : "MIGRATE"
    automatic_restart   = true
    preemptible         = false
  }

  metadata = {
    # GCE startup-script runs once per boot. Re-applying Terraform with
    # a changed script triggers an instance update; for live edits use
    # `gcloud compute instances add-metadata --metadata-from-file
    # startup-script=...` then reboot.
    startup-script = local.startup_script
    # Disable the legacy v0.1 metadata endpoint — only v1 is needed and
    # it removes a low-but-real SSRF surface.
    "block-project-ssh-keys" = "TRUE"
  }

  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }

  depends_on = [
    google_secret_manager_secret_version.agent_api_key,
    google_secret_manager_secret_version.tenant_id,
    google_secret_manager_secret_iam_member.agent_api_key_accessor,
    google_secret_manager_secret_iam_member.tenant_id_accessor,
  ]
}
