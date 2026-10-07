#!/bin/bash

# Unified GPU Cost Optimization Agent Start Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="config.yaml"
SERVICE_NAME="gpu-cost-optimizer"

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

check_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        log_info "Run './scripts/configure.sh' to create configuration"
        exit 1
    fi
    
    log_success "Configuration file found: $CONFIG_FILE"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker is not running"
        exit 1
    fi
    
    log_success "Docker is available and running"
}

check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    log_success "Docker Compose is available"
}

start_with_docker() {
    log_info "Starting agent with Docker..."
    
    # Check if container is already running
    if docker ps --filter "name=unified-gpu-agent" --format "table {{.Names}}" | grep -q "unified-gpu-agent"; then
        log_warning "Agent container is already running"
        read -p "Do you want to restart it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker stop unified-gpu-agent
            docker rm unified-gpu-agent
        else
            log_info "Keeping existing container running"
            return
        fi
    fi
    
    # Create necessary directories
    mkdir -p data logs
    
    # Start container
    docker run -d \
        --name unified-gpu-agent \
        --restart unless-stopped \
        -v "$(pwd)/$CONFIG_FILE:/app/config.yaml:ro" \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$HOME/.kube/config:/root/.kube/config:ro" \
        -p 8000:8000 \
        unified-gpu-agent:latest
    
    log_success "Agent started with Docker"
}

start_with_docker_compose() {
    log_info "Starting agent with Docker Compose..."
    
    # Check if docker-compose.yml exists
    if [[ ! -f "docker-compose.yml" ]]; then
        log_error "docker-compose.yml not found"
        exit 1
    fi
    
    # Start services
    docker-compose up -d
    
    log_success "Agent started with Docker Compose"
}

start_with_systemd() {
    log_info "Starting agent with systemd..."
    
    # Check if service exists
    if ! systemctl list-unit-files | grep -q "$SERVICE_NAME.service"; then
        log_error "Systemd service not found: $SERVICE_NAME"
        log_info "Install the systemd unit manually (see deploy/) before using --systemd"
        exit 1
    fi
    
    # Start service
    systemctl start "$SERVICE_NAME"
    
    # Check if service started successfully
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_success "Agent started with systemd"
    else
        log_error "Failed to start systemd service"
        systemctl status "$SERVICE_NAME" --no-pager
        exit 1
    fi
}

start_manually() {
    log_info "Starting agent manually..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check if main.py exists
    if [[ ! -f "main.py" ]]; then
        log_error "main.py not found"
        exit 1
    fi
    
    # Create necessary directories
    mkdir -p data logs
    
    # Start agent
    python3 main.py --config "$CONFIG_FILE" &
    AGENT_PID=$!
    
    # Save PID
    echo $AGENT_PID > gpu-agent.pid
    
    log_success "Agent started manually (PID: $AGENT_PID)"
}

show_status() {
    log_info "Agent Status:"
    
    # Check Docker container
    if docker ps --filter "name=unified-gpu-agent" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q "unified-gpu-agent"; then
        echo "Docker Container:"
        docker ps --filter "name=unified-gpu-agent" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    fi
    
    # Check systemd service
    if systemctl list-unit-files | grep -q "$SERVICE_NAME.service"; then
        echo "Systemd Service:"
        systemctl status "$SERVICE_NAME" --no-pager -l
    fi
    
    # Check manual process
    if [[ -f "gpu-agent.pid" ]]; then
        PID=$(cat gpu-agent.pid)
        if ps -p $PID > /dev/null 2>&1; then
            echo "Manual Process:"
            echo "PID: $PID"
            ps -p $PID -o pid,ppid,cmd
        fi
    fi
    
    # Check health endpoint
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Agent health check passed"
    else
        log_warning "Agent health check failed"
    fi
}

show_logs() {
    log_info "Recent logs:"
    
    # Docker logs
    if docker ps --filter "name=unified-gpu-agent" --format "{{.Names}}" | grep -q "unified-gpu-agent"; then
        echo "Docker Container Logs:"
        docker logs --tail 20 unified-gpu-agent
    fi
    
    # Systemd logs
    if systemctl list-unit-files | grep -q "$SERVICE_NAME.service"; then
        echo "Systemd Service Logs:"
        journalctl -u "$SERVICE_NAME" --no-pager -n 20
    fi
    
    # Manual process logs
    if [[ -f "logs/unified_agent.log" ]]; then
        echo "Log File:"
        tail -20 logs/unified_agent.log
    fi
}

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -d, --docker          Start with Docker"
    echo "  -c, --compose         Start with Docker Compose"
    echo "  -s, --systemd         Start with systemd"
    echo "  -m, --manual          Start manually"
    echo "  --status              Show agent status"
    echo "  --logs                Show recent logs"
    echo "  -h, --help            Show this help message"
    echo
    echo "Examples:"
    echo "  $0 --docker           Start with Docker"
    echo "  $0 --compose          Start with Docker Compose"
    echo "  $0 --systemd          Start with systemd"
    echo "  $0 --status           Show status"
    echo "  $0 --logs             Show logs"
}

# Parse command line arguments
START_METHOD=""
SHOW_STATUS=false
SHOW_LOGS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--docker)
            START_METHOD="docker"
            shift
            ;;
        -c|--compose)
            START_METHOD="compose"
            shift
            ;;
        -s|--systemd)
            START_METHOD="systemd"
            shift
            ;;
        -m|--manual)
            START_METHOD="manual"
            shift
            ;;
        --status)
            SHOW_STATUS=true
            shift
            ;;
        --logs)
            SHOW_LOGS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    if [[ "$SHOW_STATUS" == "true" ]]; then
        show_status
        exit 0
    fi
    
    if [[ "$SHOW_LOGS" == "true" ]]; then
        show_logs
        exit 0
    fi
    
    # Check configuration
    check_config
    
    # Determine start method
    if [[ -z "$START_METHOD" ]]; then
        # Auto-detect start method
        if [[ -f "docker-compose.yml" ]]; then
            START_METHOD="compose"
        elif systemctl list-unit-files | grep -q "$SERVICE_NAME.service"; then
            START_METHOD="systemd"
        elif command -v docker &> /dev/null; then
            START_METHOD="docker"
        else
            START_METHOD="manual"
        fi
        
        log_info "Auto-detected start method: $START_METHOD"
    fi
    
    # Start agent based on method
    case $START_METHOD in
        docker)
            check_docker
            start_with_docker
            ;;
        compose)
            check_docker
            check_docker_compose
            start_with_docker_compose
            ;;
        systemd)
            start_with_systemd
            ;;
        manual)
            start_manually
            ;;
        *)
            log_error "Invalid start method: $START_METHOD"
            exit 1
            ;;
    esac
    
    # Show status
    sleep 2
    show_status
    
    log_success "Agent started successfully!"
    log_info "Use '$0 --status' to check status"
    log_info "Use '$0 --logs' to view logs"
}

# Run main function
main "$@"
