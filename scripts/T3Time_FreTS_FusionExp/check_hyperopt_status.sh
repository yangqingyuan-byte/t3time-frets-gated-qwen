#!/bin/bash
# 检查参数寻优脚本的运行状态

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_FusionExp/ETTh1"
RESULT_LOG="/root/0/T3Time/experiment_results.log"

echo "=========================================="
echo "参数寻优脚本运行状态检查"
echo "=========================================="
echo ""

# 检查日志文件数量
total_logs=$(ls -1 "${LOG_DIR}"/*.log 2>/dev/null | wc -l)
echo "日志文件数量: ${total_logs}"

# 检查最近修改的日志
echo ""
echo "最近修改的日志文件（最后5个）:"
ls -lt "${LOG_DIR}"/*.log 2>/dev/null | head -5 | awk '{print "  " $9 " (修改时间: " $6 " " $7 " " $8 ")"}'

# 检查是否有失败的训练
echo ""
echo "检查失败的训练（包含 Error/Exception）:"
failed_logs=$(grep -l -E "Error|Exception|Traceback|Killed|OOM|CUDA.*error" "${LOG_DIR}"/*.log 2>/dev/null | wc -l)
echo "  失败日志数量: ${failed_logs}"

if [ "${failed_logs}" -gt 0 ]; then
  echo ""
  echo "失败的日志文件:"
  grep -l -E "Error|Exception|Traceback|Killed|OOM|CUDA.*error" "${LOG_DIR}"/*.log 2>/dev/null | head -5 | while read log; do
    echo "  ${log}"
    echo "    最后几行:"
    tail -3 "${log}" | sed 's/^/      /'
  done
fi

# 检查结果日志中的记录数
echo ""
echo "结果日志中的记录数:"
total_results=$(grep -c "T3Time_FreTS_FusionExp" "${RESULT_LOG}" 2>/dev/null || echo "0")
echo "  总记录数: ${total_results}"

# 按配置统计
echo ""
echo "按 pred_len 统计:"
for pred in 96 192 336 720; do
  count=$(grep "\"pred_len\": ${pred}" "${RESULT_LOG}" 2>/dev/null | grep -c "T3Time_FreTS_FusionExp" || echo "0")
  echo "  pred_len=${pred}: ${count} 条记录"
done

# 检查是否有正在运行的训练进程
echo ""
echo "检查是否有正在运行的训练进程:"
running=$(ps aux | grep -E "train_frets_gated_qwen_fusion_exp.py.*ETTh1" | grep -v grep | wc -l)
if [ "${running}" -gt 0 ]; then
  echo "  ✅ 有 ${running} 个训练进程正在运行"
  ps aux | grep -E "train_frets_gated_qwen_fusion_exp.py.*ETTh1" | grep -v grep | head -3
else
  echo "  ⚠️  没有正在运行的训练进程"
fi

echo ""
echo "=========================================="
