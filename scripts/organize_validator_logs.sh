#!/bin/bash
#
# organize_validator_logs.sh - Daily validator log organization
#
# Extracts yesterday's logs from PM2 and organizes them into dated files
# with separate error logs for easy debugging.
#
# Setup:
#   1. Copy logging.conf.example to logging.conf and customize paths
#   2. Add to cron: 0 0 * * * /path/to/RESI-models/scripts/organize_validator_logs.sh
#
# Output:
#   logs/validator/logs/validator_logs_DD_MM_YYYY
#   logs/validator/errors/validator_errors_DD_MM_YYYY
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/logging.conf}"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

load_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "Error: Config file not found at $CONFIG_FILE"
        echo ""
        echo "Setup required:"
        echo "  cp $SCRIPT_DIR/logging.conf.example $SCRIPT_DIR/logging.conf"
        echo "  nano $SCRIPT_DIR/logging.conf  # customize paths"
        exit 1
    fi
    source "$CONFIG_FILE"

    # Validate required variables
    if [[ -z "$LOG_BASE_DIR" ]]; then
        echo "Error: LOG_BASE_DIR not set in $CONFIG_FILE"
        exit 1
    fi
    if [[ -z "$PM2_LOG_DIR" ]]; then
        echo "Error: PM2_LOG_DIR not set in $CONFIG_FILE"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Log Organization
# -----------------------------------------------------------------------------

organize_logs() {
    local yesterday=$(date -d "yesterday" +%d_%m_%Y)
    local yesterday_pattern=$(date -d "yesterday" +%Y-%m-%d)

    local logs_dir="$LOG_BASE_DIR/validator/logs"
    local errors_dir="$LOG_BASE_DIR/validator/errors"

    # Create output directories
    mkdir -p "$logs_dir" "$errors_dir"

    # PM2 log files
    local pm2_out="$PM2_LOG_DIR/resi-validator-out.log"
    local pm2_err="$PM2_LOG_DIR/resi-validator-error.log"

    # Output files
    local log_file="$logs_dir/validator_logs_$yesterday"
    local error_file="$errors_dir/validator_errors_$yesterday"

    echo "$(date): Organizing validator logs for $yesterday"

    if [[ ! -f "$pm2_out" ]] && [[ ! -f "$pm2_err" ]]; then
        echo "  Warning: No PM2 log files found at $PM2_LOG_DIR"
        echo "  Expected: resi-validator-out.log, resi-validator-error.log"
        return 0
    fi

    # Extract yesterday's logs (all levels)
    # Log format: 2026-01-27 10:30:45 | INFO | ...
    cat "$pm2_out" "$pm2_err" 2>/dev/null | \
        grep "^$yesterday_pattern" > "$log_file" || true

    # Extract errors only (ERROR and CRITICAL levels)
    cat "$pm2_out" "$pm2_err" 2>/dev/null | \
        grep "^$yesterday_pattern" | \
        grep -E "\| ERROR \||\| CRITICAL \|" > "$error_file" || true

    # Report results and clean up empty files
    if [[ -s "$log_file" ]]; then
        local line_count=$(wc -l < "$log_file")
        echo "  Logs: $line_count lines -> $log_file"
    else
        rm -f "$log_file"
        echo "  Logs: No logs for $yesterday_pattern"
    fi

    if [[ -s "$error_file" ]]; then
        local error_count=$(wc -l < "$error_file")
        echo "  Errors: $error_count lines -> $error_file"
    else
        rm -f "$error_file"
        echo "  Errors: No errors for $yesterday_pattern"
    fi
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    load_config
    organize_logs
    echo "$(date): Complete"
}

main "$@"
