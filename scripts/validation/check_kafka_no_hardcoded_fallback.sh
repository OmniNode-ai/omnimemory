#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# OMN-4860: Reject hardcoded Kafka broker address fallbacks in omnimemory
# Mirrors the omnibase_infra hook (OMN-3554) for ecosystem parity.
#
# Suppressions:
#   # kafka-fallback-ok         — intentional test fixture default
#   # noqa                      — general suppression
#   # onex-allow-internal-ip    — intentional private-IP reference

set -euo pipefail

FAILED=0

# R1: os.getenv("KAFKA_*", non-empty) pattern in src/
MATCHES=$(grep -rn --include="*.py" \
    --exclude-dir=".venv" \
    --exclude-dir="node_modules" \
    -E "os\.getenv\([[:space:]]*[\"']KAFKA_[^\"']+[\"'][[:space:]]*,[[:space:]]*[\"'][^\"']+[\"']" \
    src/ 2>/dev/null | \
    grep -v "# kafka-fallback-ok" | \
    grep -v "# noqa" || true)

if [ -n "$MATCHES" ]; then
    echo "ERROR: Hardcoded Kafka bootstrap fallback detected in src/:"
    echo "$MATCHES"
    echo ""
    echo "FIX: Replace os.getenv(\"KAFKA_...\", \"fallback\") with:"
    echo "  os.environ[\"KAFKA_BOOTSTRAP_SERVERS\"]  # fails loudly when unset"
    echo "  os.getenv(\"KAFKA_BOOTSTRAP_SERVERS\")   # returns None when unset"
    echo "  If intentional (test fixture): add # kafka-fallback-ok"
    FAILED=1
fi

# R2: Private-IP Kafka broker addresses (Kafka-specific ports only) in src/
IP_MATCHES=$(grep -rn --include="*.py" \
    --exclude-dir=".venv" \
    --exclude-dir="node_modules" \
    -E "192\.168\.[0-9]+\.[0-9]+:(9092|19092|29092|29093)" \
    src/ 2>/dev/null | \
    grep -v "# kafka-fallback-ok" | \
    grep -v "# noqa" | \
    grep -v "# onex-allow-internal-ip" || true)

if [ -n "$IP_MATCHES" ]; then
    echo "ERROR: Hardcoded private-IP Kafka broker address in src/:"
    echo "$IP_MATCHES"
    echo ""
    echo "FIX: Use KAFKA_BOOTSTRAP_SERVERS env var."
    echo "  If intentional (test fixture): add # kafka-fallback-ok"
    FAILED=1
fi

# R3: localhost:9092 hardcoded in src/ (the canonical bad pattern)
LOCALHOST_MATCHES=$(grep -rn --include="*.py" \
    --exclude-dir=".venv" \
    --exclude-dir="node_modules" \
    -E "localhost:9092" \
    src/ 2>/dev/null | \
    grep -v "# kafka-fallback-ok" | \
    grep -v "# noqa" || true)

if [ -n "$LOCALHOST_MATCHES" ]; then
    echo "ERROR: Hardcoded localhost:9092 Kafka broker in src/:"
    echo "$LOCALHOST_MATCHES"
    echo ""
    echo "FIX: Use KAFKA_BOOTSTRAP_SERVERS env var instead of localhost:9092."
    echo "  Docker services use redpanda:9092 (internal DNS)."
    echo "  Host scripts use localhost:19092 (external port)."
    echo "  If intentional: add # kafka-fallback-ok"
    FAILED=1
fi

exit $FAILED
