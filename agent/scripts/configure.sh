#!/bin/bash

# Unified GPU Cost Optimization Agent Configuration Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="config.yaml"
CONFIG_TEMPLATE="config.yaml.example"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_config_template() {
    if [[ ! -f "$CONFIG_TEMPLATE" ]]; then
        log_error "Configuration template not found: $CONFIG_TEMPLATE"
        exit 1
    fi
}

create_config_file() {
    if [[ -f "$CONFIG_FILE" ]]; then
        log_warning "Configuration file already exists: $CONFIG_FILE"
        read -p "Do you want to overwrite it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Keeping existing configuration file"
            return
        fi
    fi
    
    log_info "Creating configuration file from template..."
    cp "$CONFIG_TEMPLATE" "$CONFIG_FILE"
    log_success "Configuration file created: $CONFIG_FILE"
}

configure_cloud_providers() {
    log_info "Configuring cloud providers..."
    
    # AWS Configuration
    read -p "Enable AWS monitoring? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        AWS_ENABLED=true
        read -p "AWS Region (default: us-east-1): " AWS_REGION
        AWS_REGION=${AWS_REGION:-us-east-1}
        
        read -p "AWS Services to monitor (default: ec2,sagemaker): " AWS_SERVICES
        AWS_SERVICES=${AWS_SERVICES:-ec2,sagemaker}
        
        log_success "AWS monitoring configured"
    else
        AWS_ENABLED=false
        log_info "AWS monitoring disabled"
    fi
    
    # Azure Configuration
    read -p "Enable Azure monitoring? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        AZURE_ENABLED=true
        read -p "Azure Subscription ID: " AZURE_SUBSCRIPTION_ID
        
        if [[ -z "$AZURE_SUBSCRIPTION_ID" ]]; then
            log_error "Azure Subscription ID is required"
            exit 1
        fi
        
        read -p "Azure Resource Group (optional): " AZURE_RESOURCE_GROUP
        
        log_success "Azure monitoring configured"
    else
        AZURE_ENABLED=false
        log_info "Azure monitoring disabled"
    fi
    
    # Google Cloud Configuration
    read -p "Enable Google Cloud monitoring? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        GCP_ENABLED=true
        read -p "Google Cloud Project ID: " GCP_PROJECT_ID
        
        if [[ -z "$GCP_PROJECT_ID" ]]; then
            log_error "Google Cloud Project ID is required"
            exit 1
        fi
        
        read -p "GCP Region (default: us-central1): " GCP_REGION
        GCP_REGION=${GCP_REGION:-us-central1}
        
        log_success "Google Cloud monitoring configured"
    else
        GCP_ENABLED=false
        log_info "Google Cloud monitoring disabled"
    fi
    
    # Kubernetes Configuration
    read -p "Enable Kubernetes monitoring? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        K8S_ENABLED=true
        read -p "Kubernetes Config Path (default: ~/.kube/config): " K8S_CONFIG_PATH
        K8S_CONFIG_PATH=${K8S_CONFIG_PATH:-~/.kube/config}
        
        read -p "Kubernetes Namespaces (default: default): " K8S_NAMESPACES
        K8S_NAMESPACES=${K8S_NAMESPACES:-default}
        
        log_success "Kubernetes monitoring configured"
    else
        K8S_ENABLED=false
        log_info "Kubernetes monitoring disabled"
    fi
    
    # SageMaker Configuration
    read -p "Enable SageMaker monitoring? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        SAGEMAKER_ENABLED=true
        read -p "SageMaker Region (default: us-east-1): " SAGEMAKER_REGION
        SAGEMAKER_REGION=${SAGEMAKER_REGION:-us-east-1}
        
        read -p "SageMaker Services (default: training_jobs,endpoints): " SAGEMAKER_SERVICES
        SAGEMAKER_SERVICES=${SAGEMAKER_SERVICES:-training_jobs,endpoints}
        
        log_success "SageMaker monitoring configured"
    else
        SAGEMAKER_ENABLED=false
        log_info "SageMaker monitoring disabled"
    fi
}

configure_dashboard() {
    log_info "Configuring dashboard connection..."
    
    read -p "Dashboard API URL (for SaaS mode): " DASHBOARD_API_URL
    read -p "Dashboard API Key: " DASHBOARD_API_KEY
    
    if [[ -n "$DASHBOARD_API_URL" && -n "$DASHBOARD_API_KEY" ]]; then
        DASHBOARD_SYNC_ENABLED=true
        log_success "Dashboard connection configured"
    else
        DASHBOARD_SYNC_ENABLED=false
        log_warning "Dashboard sync disabled - running in local mode"
    fi
}

configure_alerting() {
    log_info "Configuring alerting..."
    
    read -p "Slack Webhook URL (optional): " SLACK_WEBHOOK
    read -p "Email for alerts (optional): " EMAIL_TO
    
    if [[ -n "$SLACK_WEBHOOK" || -n "$EMAIL_TO" ]]; then
        log_success "Alerting configured"
    else
        log_warning "No alerting configured"
    fi
}

configure_monitoring() {
    log_info "Configuring monitoring settings..."
    
    read -p "Monitoring interval in seconds (default: 300): " MONITORING_INTERVAL
    MONITORING_INTERVAL=${MONITORING_INTERVAL:-300}
    
    read -p "Idle GPU threshold in minutes (default: 10): " IDLE_THRESHOLD
    IDLE_THRESHOLD=${IDLE_THRESHOLD:-10}
    
    read -p "Daily cost threshold in USD (default: 100): " COST_THRESHOLD
    COST_THRESHOLD=${COST_THRESHOLD:-100}
    
    log_success "Monitoring settings configured"
}

update_config_file() {
    log_info "Updating configuration file..."
    
    # Create temporary file
    TEMP_CONFIG=$(mktemp)
    
    # Update configuration based on user input
    cat > "$TEMP_CONFIG" << EOF
# Unified GPU Cost Optimization Agent Configuration

monitoring:
  enabled_clouds:
EOF
    
    # Add enabled clouds
    if [[ "$AWS_ENABLED" == "true" ]]; then
        echo "    - aws" >> "$TEMP_CONFIG"
    fi
    if [[ "$AZURE_ENABLED" == "true" ]]; then
        echo "    - azure" >> "$TEMP_CONFIG"
    fi
    if [[ "$GCP_ENABLED" == "true" ]]; then
        echo "    - gcp" >> "$TEMP_CONFIG"
    fi
    if [[ "$K8S_ENABLED" == "true" ]]; then
        echo "    - kubernetes" >> "$TEMP_CONFIG"
    fi
    if [[ "$SAGEMAKER_ENABLED" == "true" ]]; then
        echo "    - sagemaker" >> "$TEMP_CONFIG"
    fi
    
    cat >> "$TEMP_CONFIG" << EOF
  
  interval: $MONITORING_INTERVAL
  
  aws:
    enabled: $AWS_ENABLED
    region: ${AWS_REGION:-us-east-1}
    services:
EOF
    
    if [[ "$AWS_ENABLED" == "true" ]]; then
        IFS=',' read -ra SERVICES <<< "$AWS_SERVICES"
        for service in "${SERVICES[@]}"; do
            echo "      - $service" >> "$TEMP_CONFIG"
        done
    fi
    
    cat >> "$TEMP_CONFIG" << EOF
  
  azure:
    enabled: $AZURE_ENABLED
    subscription_id: "${AZURE_SUBSCRIPTION_ID:-}"
    resource_group: "${AZURE_RESOURCE_GROUP:-}"
  
  gcp:
    enabled: $GCP_ENABLED
    project_id: "${GCP_PROJECT_ID:-}"
    region: ${GCP_REGION:-us-central1}
  
  kubernetes:
    enabled: $K8S_ENABLED
    kubeconfig_path: "${K8S_CONFIG_PATH:-~/.kube/config}"
    namespaces:
EOF
    
    if [[ "$K8S_ENABLED" == "true" ]]; then
        IFS=',' read -ra NAMESPACES <<< "$K8S_NAMESPACES"
        for namespace in "${NAMESPACES[@]}"; do
            echo "      - $namespace" >> "$TEMP_CONFIG"
        done
    fi
    
    cat >> "$TEMP_CONFIG" << EOF
  
  sagemaker:
    enabled: $SAGEMAKER_ENABLED
    region: ${SAGEMAKER_REGION:-us-east-1}
    services:
EOF
    
    if [[ "$SAGEMAKER_ENABLED" == "true" ]]; then
        IFS=',' read -ra SERVICES <<< "$SAGEMAKER_SERVICES"
        for service in "${SERVICES[@]}"; do
            echo "      - $service" >> "$TEMP_CONFIG"
        done
    fi
    
    cat >> "$TEMP_CONFIG" << EOF

dashboard:
  api_url: "${DASHBOARD_API_URL:-}"
  api_key: "${DASHBOARD_API_KEY:-}"
  sync_enabled: $DASHBOARD_SYNC_ENABLED

alerting:
  slack_webhook: "${SLACK_WEBHOOK:-}"
  email:
    to: "${EMAIL_TO:-}"
    from: "gpu-agent@yourcompany.com"
  
  thresholds:
    idle_gpu_minutes: $IDLE_THRESHOLD
    daily_cost_usd: $COST_THRESHOLD

logging:
  level: INFO
  file: "./logs/unified_agent.log"

retention:
  data_days: 30
  metrics_days: 7
  cost_days: 90
EOF
    
    # Replace original config file
    mv "$TEMP_CONFIG" "$CONFIG_FILE"
    
    log_success "Configuration file updated: $CONFIG_FILE"
}

validate_configuration() {
    log_info "Validating configuration..."
    
    # Check if at least one cloud provider is enabled
    if [[ "$AWS_ENABLED" != "true" && "$AZURE_ENABLED" != "true" && "$GCP_ENABLED" != "true" && "$K8S_ENABLED" != "true" && "$SAGEMAKER_ENABLED" != "true" ]]; then
        log_error "At least one cloud provider must be enabled"
        exit 1
    fi
    
    # Check required fields for enabled providers
    if [[ "$AZURE_ENABLED" == "true" && -z "$AZURE_SUBSCRIPTION_ID" ]]; then
        log_error "Azure Subscription ID is required when Azure monitoring is enabled"
        exit 1
    fi
    
    if [[ "$GCP_ENABLED" == "true" && -z "$GCP_PROJECT_ID" ]]; then
        log_error "Google Cloud Project ID is required when GCP monitoring is enabled"
        exit 1
    fi
    
    log_success "Configuration validation passed"
}

show_next_steps() {
    log_success "Configuration completed successfully!"
    echo
    log_info "Next steps:"
    echo "1. Set up cloud credentials:"
    if [[ "$AWS_ENABLED" == "true" ]]; then
        echo "   - AWS: aws configure (or use IAM role)"
    fi
    if [[ "$AZURE_ENABLED" == "true" ]]; then
        echo "   - Azure: az login (or use managed identity)"
    fi
    if [[ "$GCP_ENABLED" == "true" ]]; then
        echo "   - Google Cloud: gcloud auth application-default login"
    fi
    if [[ "$K8S_ENABLED" == "true" ]]; then
        echo "   - Kubernetes: kubectl config use-context your-cluster"
    fi
    echo
    echo "2. Start the agent:"
    echo "   - Docker: docker-compose up -d"
    echo "   - Systemd: systemctl start gpu-cost-optimizer"
    echo "   - Manual: python main.py --config config.yaml"
    echo
    echo "3. Check agent status:"
    echo "   - gpu-agent status"
    echo "   - gpu-agent health"
    echo
    log_info "Configuration file: $CONFIG_FILE"
}

# Main configuration process
main() {
    log_info "Starting GPU Cost Optimization Agent configuration..."
    
    check_config_template
    create_config_file
    configure_cloud_providers
    configure_dashboard
    configure_alerting
    configure_monitoring
    update_config_file
    validate_configuration
    show_next_steps
}

# Run main function
main "$@"
