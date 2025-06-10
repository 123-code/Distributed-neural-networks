#!/bin/bash

# Script para iniciar un worker del sistema distribuido Qwen
# Uso: ./start_worker.sh <rank> [world_size] [master_addr] [master_port]

RANK=${1:-0}
WORLD_SIZE=${2:-3}
MASTER_ADDR=${3:-localhost}
MASTER_PORT=${4:-29500}

echo "Iniciando worker $RANK de $WORLD_SIZE procesos totales..."
echo "Master: $MASTER_ADDR:$MASTER_PORT"

python worker.py \
    --rank $RANK \
    --world_size $WORLD_SIZE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT 