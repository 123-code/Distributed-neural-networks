#!/bin/bash

# Script para iniciar el coordinador del sistema distribuido Qwen
# Uso: ./start_coordinator.sh [world_size] [master_addr] [master_port]

WORLD_SIZE=${1:-3}
MASTER_ADDR=${2:-localhost}
MASTER_PORT=${3:-29500}

echo "Iniciando coordinador para sistema con $WORLD_SIZE procesos..."
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# El coordinador es siempre el último rank
COORDINATOR_RANK=$((WORLD_SIZE - 1))

python coordinator.py \
    --rank $COORDINATOR_RANK \
    --world_size $WORLD_SIZE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT 