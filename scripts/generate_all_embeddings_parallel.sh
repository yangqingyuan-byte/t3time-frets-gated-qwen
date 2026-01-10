#!/bin/bash
# 并行生成所有数据集的 qwen3_0.6b 嵌入文件
# 自动检测 dataset 目录下的所有 CSV 文件，并在不同 GPU 上并行运行
# 用法: bash scripts/generate_all_embeddings_parallel.sh

set -e

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 切换到项目根目录
cd "${PROJECT_ROOT}"

# 激活 conda 环境（如果需要）
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

# 设置 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH-}"

# 数据集目录
DATASET_DIR="${PROJECT_ROOT}/dataset"

echo "=========================================="
echo "并行生成所有数据集的 Qwen3-0.6B 嵌入文件"
echo "=========================================="
echo ""

# 检查数据集目录是否存在
if [ ! -d "${DATASET_DIR}" ]; then
    echo "❌ 错误: 数据集目录不存在: ${DATASET_DIR}"
    exit 1
fi

# 获取所有 CSV 文件（去除 .csv 后缀）
echo "🔍 扫描数据集目录: ${DATASET_DIR}"
datasets=()
for csv_file in "${DATASET_DIR}"/*.csv; do
    if [ -f "${csv_file}" ]; then
        dataset_name=$(basename "${csv_file}" .csv)
        datasets+=("${dataset_name}")
        echo "  ✓ 发现数据集: ${dataset_name}"
    fi
done

if [ ${#datasets[@]} -eq 0 ]; then
    echo "❌ 错误: 未找到任何 CSV 数据集文件"
    exit 1
fi

echo ""
echo "📊 共找到 ${#datasets[@]} 个数据集: ${datasets[*]}"
echo ""

# GPU 数量（8卡）
NUM_GPUS=8

# 需要跳过的数据集列表（已存在嵌入，不需要重新生成）
SKIP_DATASETS=("ETTh1")

# 检查每个数据集是否已经有嵌入文件
echo "🔍 检查已存在的嵌入文件..."
datasets_to_process=()
for dataset in "${datasets[@]}"; do
    # 跳过指定的数据集
    skip=false
    for skip_dataset in "${SKIP_DATASETS[@]}"; do
        if [ "${dataset}" == "${skip_dataset}" ]; then
            echo "  ⏭️  跳过 ${dataset} (已在跳过列表中)"
            skip=true
            break
        fi
    done
    if [ "$skip" = true ]; then
        continue
    fi
    embed_dir="${PROJECT_ROOT}/Embeddings/${dataset}/qwen3_0.6b"
    if [ -d "${embed_dir}/train" ] && [ -d "${embed_dir}/val" ] && [ -d "${embed_dir}/test" ]; then
        train_count=$(find "${embed_dir}/train" -name "*.h5" 2>/dev/null | wc -l)
        val_count=$(find "${embed_dir}/val" -name "*.h5" 2>/dev/null | wc -l)
        test_count=$(find "${embed_dir}/test" -name "*.h5" 2>/dev/null | wc -l)
        
        if [ "${train_count}" -gt 0 ] && [ "${val_count}" -gt 0 ] && [ "${test_count}" -gt 0 ]; then
            echo "  ⏭️  跳过 ${dataset} (嵌入已存在: train=${train_count}, val=${val_count}, test=${test_count})"
        else
            echo "  ⚠️  ${dataset} 嵌入不完整，将重新生成"
            datasets_to_process+=("${dataset}")
        fi
    else
        echo "  ➕ ${dataset} 需要生成嵌入"
        datasets_to_process+=("${dataset}")
    fi
done

if [ ${#datasets_to_process[@]} -eq 0 ]; then
    echo ""
    echo "✅ 所有数据集的嵌入文件已存在，无需重新生成！"
    exit 0
fi

echo ""
echo "📋 需要处理的数据集 (${#datasets_to_process[@]} 个): ${datasets_to_process[*]}"
echo ""

# 创建日志目录
LOG_DIR="${PROJECT_ROOT}/Results/embed_generation_logs"
mkdir -p "${LOG_DIR}"

# 并行生成嵌入
echo "🚀 开始并行生成嵌入（使用 ${NUM_GPUS} 个 GPU）..."
echo ""

pids=()
gpu_assignments=()

for i in "${!datasets_to_process[@]}"; do
    dataset="${datasets_to_process[$i]}"
    gpu_id=$((i % NUM_GPUS))
    
    echo "  📌 分配 ${dataset} -> GPU ${gpu_id}"
    gpu_assignments+=("${dataset}:GPU${gpu_id}")
    
    # 为每个数据集创建日志文件
    log_file="${LOG_DIR}/${dataset}_gpu${gpu_id}.log"
    
    # 在后台运行生成脚本
    (
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始生成 ${dataset} 的嵌入 (GPU ${gpu_id})" | tee -a "${log_file}"
        
        # 生成训练集嵌入
        python storage/store_emb_qwen3_0.6b.py \
            --data_path ${dataset} \
            --divide train \
            --input_len 96 \
            --output_len 96 \
            --device cuda \
            --batch_size 1 \
            --num_workers 4 \
            --d_model 1024 \
            --l_layers 28 \
            --model_name "Qwen/Qwen3-0.6B" \
            --embed_version qwen3_0.6b \
            2>&1 | tee -a "${log_file}"
        
        # 生成验证集嵌入
        python storage/store_emb_qwen3_0.6b.py \
            --data_path ${dataset} \
            --divide val \
            --input_len 96 \
            --output_len 96 \
            --device cuda \
            --batch_size 1 \
            --num_workers 4 \
            --d_model 1024 \
            --l_layers 28 \
            --model_name "Qwen/Qwen3-0.6B" \
            --embed_version qwen3_0.6b \
            2>&1 | tee -a "${log_file}"
        
        # 生成测试集嵌入
        python storage/store_emb_qwen3_0.6b.py \
            --data_path ${dataset} \
            --divide test \
            --input_len 96 \
            --output_len 96 \
            --device cuda \
            --batch_size 1 \
            --num_workers 4 \
            --d_model 1024 \
            --l_layers 28 \
            --model_name "Qwen/Qwen3-0.6B" \
            --embed_version qwen3_0.6b \
            2>&1 | tee -a "${log_file}"
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ ${dataset} 嵌入生成完成 (GPU ${gpu_id})" | tee -a "${log_file}"
    ) &
    
    pids+=($!)
    
    # 避免同时启动太多进程，稍微延迟一下
    sleep 2
done

echo ""
echo "⏳ 所有任务已启动，等待完成..."
echo "📝 日志文件保存在: ${LOG_DIR}/"
echo ""

# 等待所有后台任务完成
failed_datasets=()
for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    dataset="${datasets_to_process[$i]}"
    
    if wait "${pid}"; then
        echo "✅ ${dataset} 完成"
    else
        echo "❌ ${dataset} 失败 (PID: ${pid})"
        failed_datasets+=("${dataset}")
    fi
done

echo ""
echo "=========================================="
if [ ${#failed_datasets[@]} -eq 0 ]; then
    echo "✅ 所有数据集的嵌入文件生成完成！"
    echo ""
    echo "📊 GPU 分配情况:"
    for assignment in "${gpu_assignments[@]}"; do
        echo "  ${assignment}"
    done
    echo ""
    echo "📁 嵌入文件保存路径: ./Embeddings/{数据集名称}/qwen3_0.6b/{train,val,test}/"
else
    echo "⚠️  部分数据集生成失败:"
    for dataset in "${failed_datasets[@]}"; do
        echo "  ❌ ${dataset}"
    done
    echo ""
    echo "请查看日志文件: ${LOG_DIR}/"
fi
echo "=========================================="
