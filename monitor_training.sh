#!/bin/bash
# 训练监控脚本 - 实时查看两个模型的训练进度

echo "=========================================="
echo "训练进度实时监控"
echo "按 Ctrl+C 退出"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo "训练进度监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    echo "【原始 T3Time (仅 FFT)】"
    echo "----------------------------------------"
    if [ -f "logs_original_training.log" ]; then
        # 提取最新的 epoch 和 loss
        latest_epoch=$(grep -E "^Epoch:" logs_original_training.log | tail -1 | grep -oE "Epoch: [0-9]+" | grep -oE "[0-9]+")
        latest_train_loss=$(grep -E "Train Loss:" logs_original_training.log | tail -1 | grep -oE "Train Loss: [0-9.]+" | grep -oE "[0-9.]+")
        latest_valid_loss=$(grep -E "Valid Loss:" logs_original_training.log | tail -1 | grep -oE "Valid Loss: [0-9.]+" | grep -oE "[0-9.]+")
        latest_test_mse=$(grep -E "Test MSE:" logs_original_training.log | tail -1 | grep -oE "Test MSE: [0-9.]+" | grep -oE "[0-9.]+")
        
        if [ ! -z "$latest_epoch" ]; then
            echo "当前 Epoch: $latest_epoch"
            [ ! -z "$latest_train_loss" ] && echo "Train Loss: $latest_train_loss"
            [ ! -z "$latest_valid_loss" ] && echo "Valid Loss: $latest_valid_loss"
            [ ! -z "$latest_test_mse" ] && echo "Test MSE: $latest_test_mse"
        else
            echo "训练中..."
        fi
        echo ""
        echo "最新日志（最后3行）："
        tail -3 logs_original_training.log | sed 's/^/  /'
    else
        echo "日志文件不存在"
    fi
    
    echo ""
    echo "【FFT + VMD 模型】"
    echo "----------------------------------------"
    if [ -f "logs_fft_vmd_training.log" ]; then
        # 提取最新的 epoch 和 loss
        latest_epoch=$(grep -E "^Epoch:" logs_fft_vmd_training.log | tail -1 | grep -oE "Epoch: [0-9]+" | grep -oE "[0-9]+")
        latest_train_loss=$(grep -E "Train Loss:" logs_fft_vmd_training.log | tail -1 | grep -oE "Train Loss: [0-9.]+" | grep -oE "[0-9.]+")
        latest_valid_loss=$(grep -E "Valid Loss:" logs_fft_vmd_training.log | tail -1 | grep -oE "Valid Loss: [0-9.]+" | grep -oE "[0-9.]+")
        latest_test_mse=$(grep -E "Test MSE:" logs_fft_vmd_training.log | tail -1 | grep -oE "Test MSE: [0-9.]+" | grep -oE "[0-9.]+")
        
        if [ ! -z "$latest_epoch" ]; then
            echo "当前 Epoch: $latest_epoch"
            [ ! -z "$latest_train_loss" ] && echo "Train Loss: $latest_train_loss"
            [ ! -z "$latest_valid_loss" ] && echo "Valid Loss: $latest_valid_loss"
            [ ! -z "$latest_test_mse" ] && echo "Test MSE: $latest_test_mse"
        else
            echo "训练中..."
        fi
        echo ""
        echo "最新日志（最后3行）："
        tail -3 logs_fft_vmd_training.log | sed 's/^/  /'
    else
        echo "日志文件不存在"
    fi
    
    echo ""
    echo "=========================================="
    echo "进程状态:"
    ps aux | grep -E "(train.py|train_fft_vmd.py)" | grep -v grep | wc -l | xargs echo "  运行中的训练进程数:"
    echo ""
    echo "下次更新: 30秒后 (按 Ctrl+C 退出)"
    sleep 30
done

