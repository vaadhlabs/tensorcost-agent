/**
 * gke-workload — run the unified-gpu-agent on an EXISTING GKE cluster
 * with Workload Identity.
 *
 * Trust model — what makes this module different from "kubectl apply
 * the chart with a service-account JSON":
 *
 *   1. A GCP service account is created for the agent.
 *   2. A Kubernetes ServiceAccount is created in `var.namespace`.
 *   3. The K8s SA is annotated `iam.gke.io/gcp-service-account=<gcp-sa>`
 *      — GKE's Workload Identity webhook wires the pod's tokens.
 *   4. The GCP SA gets `roles/iam.workloadIdentityUser` for the
 *      principal `serviceAccount:<project>.svc.id.goog[<ns>/<k8s-sa>]`
 *      — this is the IAM binding that lets the K8s SA impersonate the
 *      GCP SA via the metadata server.
 *   5. Secret Manager secrets are scoped (accessor role on the secret
 *      itself, not the project) to the GCP SA.
 *
 * Result: the pod fetches its API key by calling Secret Manager with
 * the metadata-server-minted short-lived token. NO JSON KEY IS EVER
 * PROVISIONED. The §8.3 audit explicitly called out the Helm chart's
 * GCP path for the JSON-key anti-pattern; this module is the
 * Terraform answer.
 *
 * Pre-req on the cluster:
 *   gcloud container clusters update <cluster> \
 *     --workload-pool=<project>.svc.id.goog
 *
 * If the cluster doesn't have Workload Identity enabled, the module
 * still applies but the pod will fail Secret Manager calls with
 * `IAM_PERMISSION_DENIED` — the README spells this out.
 */

// ─── 0. Look up the cluster (used for the workload-pool reference) ──────

data "google_container_cluster" "target" {
  project  = var.project_id
  name     = var.cluster_name
  location = var.cluster_location
}

// ─── 1. GCP service account the pod impersonates ────────────────────────

resource "google_service_account" "agent" {
  project      = var.project_id
  account_id   = local.gcp_sa_account_id
  display_name = "TensorCost unified-gpu-agent on GKE (${var.name})"
  description  = "Impersonated by the K8s ServiceAccount '${var.namespace}/${local.k8s_sa_name}' via Workload Identity. No JSON keys ever issued."
}

// Workload Identity binding: the K8s SA can impersonate the GCP SA.
// The principal name is constructed exactly as GKE expects:
//   serviceAccount:<workload-pool>[<namespace>/<k8s-sa>]
resource "google_service_account_iam_member" "workload_identity_user" {
  service_account_id = google_service_account.agent.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${local.workload_pool}[${var.namespace}/${local.k8s_sa_name}]"
}

// ─── 2. Project-level IAM (minimum-viable) ──────────────────────────────

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

// ─── 3. Secrets in Secret Manager (scoped IAM) ──────────────────────────

resource "google_secret_manager_secret" "agent_api_key" {
  project   = var.project_id
  secret_id = "${var.name}-gke-agent-api-key"

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
  secret_id = "${var.name}-gke-tenant-id"

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

// ─── 4. Kubernetes objects (Namespace, SA, Deployment/DaemonSet) ────────

resource "kubernetes_namespace" "agent" {
  metadata {
    name   = var.namespace
    labels = local.common_labels
  }
}

resource "kubernetes_service_account" "agent" {
  metadata {
    name      = local.k8s_sa_name
    namespace = kubernetes_namespace.agent.metadata[0].name
    labels    = local.common_labels

    // The annotation that wires Workload Identity. GKE's webhook looks
    // for this exact key — without it, the pod's tokens won't carry
    // the GCP-SA-impersonation claim.
    annotations = {
      "iam.gke.io/gcp-service-account" = google_service_account.agent.email
    }
  }

  automount_service_account_token = true
}

locals {
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

  pod_env = [
    { name = "BACKEND_API_URL", value = var.backend_api_url },
    { name = "GRPC_TARGET", value = "${var.grpc_host}:${var.grpc_port}" },
    { name = "GRPC_USE_TLS", value = "true" },
    { name = "COMM_MODE", value = var.grpc_host == "" ? "http" : "both" },
    { name = "MONITORING_INTERVAL", value = tostring(var.metrics_push_interval) },
    { name = "ACTIVE_MONITORS", value = "gcp,nvml" },
    { name = "GCP_ENABLED", value = "true" },
    { name = "GCP_PROJECT_ID", value = var.project_id },
    { name = "AGENT_HOSTNAME", value = local.resolved_agent_host },
    { name = "SECRET_AGENT_API_KEY_NAME", value = google_secret_manager_secret.agent_api_key.secret_id },
    { name = "SECRET_TENANT_ID_NAME", value = google_secret_manager_secret.tenant_id.secret_id },
  ]

  pod_template_metadata = {
    labels = merge(local.common_labels, {
      "app.kubernetes.io/name"      = "unified-gpu-agent"
      "app.kubernetes.io/instance"  = var.name
      "app.kubernetes.io/component" = "agent"
    })
  }
}

// ─── 4a. Deployment (default) ────────────────────────────────────────────

resource "kubernetes_deployment" "agent" {
  count = var.deploy_as == "deployment" ? 1 : 0

  metadata {
    name      = var.name
    namespace = kubernetes_namespace.agent.metadata[0].name
    labels    = local.common_labels
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        "app.kubernetes.io/name"     = "unified-gpu-agent"
        "app.kubernetes.io/instance" = var.name
      }
    }

    template {
      metadata {
        labels = local.pod_template_metadata.labels
      }

      spec {
        service_account_name            = kubernetes_service_account.agent.metadata[0].name
        automount_service_account_token = true

        node_selector = var.node_selector

        dynamic "toleration" {
          for_each = var.tolerations
          content {
            key      = toleration.value.key
            operator = toleration.value.operator
            value    = toleration.value.value
            effect   = toleration.value.effect
          }
        }

        container {
          name  = "agent"
          image = local.agent_image_ref

          dynamic "env" {
            for_each = local.pod_env
            content {
              name  = env.value.name
              value = env.value.value
            }
          }

          // POD_NAME for AGENT_HOSTNAME default via downward API.
          env {
            name = "POD_NAME"
            value_from {
              field_ref {
                field_path = "metadata.name"
              }
            }
          }

          resources {
            requests = try(var.resources.requests, { cpu = "100m", memory = "128Mi" })
            limits   = try(var.resources.limits, { cpu = "500m", memory = "512Mi" })
          }

          security_context {
            allow_privilege_escalation = false
            read_only_root_filesystem  = false
            run_as_non_root            = true
            run_as_user                = 65532
            capabilities {
              drop = ["ALL"]
            }
          }
        }
      }
    }
  }

  depends_on = [
    google_service_account_iam_member.workload_identity_user,
    google_secret_manager_secret_iam_member.agent_api_key_accessor,
    google_secret_manager_secret_iam_member.tenant_id_accessor,
  ]
}

// ─── 4b. DaemonSet (one pod per matching node — GPU sampling) ───────────

resource "kubernetes_daemonset" "agent" {
  count = var.deploy_as == "daemonset" ? 1 : 0

  metadata {
    name      = var.name
    namespace = kubernetes_namespace.agent.metadata[0].name
    labels    = local.common_labels
  }

  spec {
    selector {
      match_labels = {
        "app.kubernetes.io/name"     = "unified-gpu-agent"
        "app.kubernetes.io/instance" = var.name
      }
    }

    template {
      metadata {
        labels = local.pod_template_metadata.labels
      }

      spec {
        service_account_name            = kubernetes_service_account.agent.metadata[0].name
        automount_service_account_token = true

        node_selector = var.node_selector

        dynamic "toleration" {
          for_each = var.tolerations
          content {
            key      = toleration.value.key
            operator = toleration.value.operator
            value    = toleration.value.value
            effect   = toleration.value.effect
          }
        }

        container {
          name  = "agent"
          image = local.agent_image_ref

          dynamic "env" {
            for_each = local.pod_env
            content {
              name  = env.value.name
              value = env.value.value
            }
          }

          env {
            name = "POD_NAME"
            value_from {
              field_ref {
                field_path = "metadata.name"
              }
            }
          }

          env {
            name = "NODE_NAME"
            value_from {
              field_ref {
                field_path = "spec.nodeName"
              }
            }
          }

          resources {
            requests = try(var.resources.requests, { cpu = "100m", memory = "128Mi" })
            limits   = try(var.resources.limits, { cpu = "500m", memory = "512Mi" })
          }

          security_context {
            allow_privilege_escalation = false
            run_as_non_root            = true
            run_as_user                = 65532
            capabilities {
              drop = ["ALL"]
            }
          }
        }
      }
    }
  }

  depends_on = [
    google_service_account_iam_member.workload_identity_user,
    google_secret_manager_secret_iam_member.agent_api_key_accessor,
    google_secret_manager_secret_iam_member.tenant_id_accessor,
  ]
}
