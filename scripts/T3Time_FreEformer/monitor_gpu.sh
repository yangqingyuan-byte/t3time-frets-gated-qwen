#!/bin/bash
# GPU资源监控脚本，用于判断是否可以增加并行数

while true; do
  clear
  echo "=========================================="
  echo "GPU 资源监控 - $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=========================================="
  
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits | \
  awk -F', ' '{
    mem_used = $2
    mem_total = $3
    gpu_util = $4
    mem_util = $5
    mem_free = mem_total - mem_used
    mem_free_gb = mem_free / 1024
    
    printf "GPU %d:\n", $1
    printf "  内存: %d MB / %d MB (%.1f%%) - 剩余 %.1f GB\n", mem_used, mem_total, (mem_used/mem_total)*100, mem_free_gb
    printf "  GPU利用率: %d%%\n", gpu_util
    printf "  内存利用率: %d%%\n", mem_util
    
    # 估算可并行数（假设每个实验需要1.5GB）
    estimated_parallel = int(mem_free / 1536)
    printf "  估算可并行数（按1.5GB/实验）: %d\n", estimated_parallel
    printf "\n"
  }'
  
  echo "=========================================="
  echo "建议："
  echo "  - 如果内存使用 < 50% 且 GPU利用率 < 95%，可以增加并行数"
  echo "  - 如果内存使用 > 80% 或 GPU利用率持续100%，保持当前并行数"
  echo "  - 按 Ctrl+C 退出监控"
  echo "=========================================="
  
  sleep 5
done
