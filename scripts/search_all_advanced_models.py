import subprocess
import os
import time

# --- 通用配置 ---
SEEDS = list(range(2020, 2041))  # 2020 到 2040
DATA_PATH = "ETTh1"
SEQ_LEN = 96
PRED_LEN = 96
EPOCHS = 100
ES_PATIENCE = 15
BATCH_SIZE = 16
EMBED_VERSION = "qwen3_0.6b"

# --- 核心寻优空间 (可根据计算资源调整) ---
LRS = [1e-4]
DROPOUTS = [0.4]
CHANNELS = [256]

# --- 模型特定配置 ---
MODELS_TO_SEARCH = [
    {
        "script": "train_wavelet_packet_gated_qwen.py",
        "name": "Wavelet_Packet",
        "params": {
            "--wp_level": ["2", "3"] # 可选: ["2", "3"]
        }
    },
    # {
    #     "script": "train_wavelet_gated_shape_qwen.py",
    #     "name": "Wavelet_Shape",
    #     "params": {
    #         "--shape_lambda": ["0.1", "0.05"] 
    #     }
    # },
    # {
    #     "script": "train_inception_swiglu_gated_qwen.py",
    #     "name": "Inception_SwiGLU",
    #     "params": {
    #         "--d_ff": ["128", "256"]
    #     }
    # }
]

def run_cmd(cmd):
    print(f"\n执行命令: {' '.join(cmd)}")
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        print(f"完成! 耗时: {duration/60:.2f} 分钟")
    except subprocess.CalledProcessError as e:
        print(f"!!! 运行失败: {e}")

def main():
    if not os.path.exists("train.py"):
        print("错误: 请在项目根目录下运行此脚本")
        return

    total_configs = len(MODELS_TO_SEARCH) * len(SEEDS) * len(LRS) * len(DROPOUTS) * len(CHANNELS)
    # 还要乘以每个模型特有的参数组合数
    actual_total = 0
    for m in MODELS_TO_SEARCH:
        m_params_count = 1
        for p_vals in m["params"].values():
            m_params_count *= len(p_vals)
        actual_total += len(SEEDS) * len(LRS) * len(DROPOUTS) * len(CHANNELS) * m_params_count

    count = 0
    print(f"开始综合参数寻优，预计总实验数: {actual_total}")

    for model_cfg in MODELS_TO_SEARCH:
        script = model_cfg["script"]
        m_name = model_cfg["name"]
        
        # 获取该模型特有的参数列表
        spec_keys = list(model_cfg["params"].keys())
        spec_values = list(model_cfg["params"].values())
        
        # 处理模型特定参数的组合（简单实现，支持多组参数）
        import itertools
        for spec_comb in itertools.product(*spec_values):
            spec_dict = dict(zip(spec_keys, spec_comb))
            
            for seed in SEEDS:
                for lr in LRS:
                    for dropout in DROPOUTS:
                        for channel in CHANNELS:
                            count += 1
                            print(f"\n[{count}/{actual_total}] 模型: {m_name} | Seed: {seed} | LR: {lr} | DP: {dropout} | CH: {channel}")
                            
                            cmd = [
                                "python", script,
                                "--data_path", DATA_PATH,
                                "--seq_len", str(SEQ_LEN),
                                "--pred_len", str(PRED_LEN),
                                "--batch_size", str(BATCH_SIZE),
                                "--learning_rate", str(lr),
                                "--dropout_n", str(dropout),
                                "--channel", str(channel),
                                "--seed", str(seed),
                                "--epochs", str(EPOCHS),
                                "--es_patience", str(ES_PATIENCE),
                                "--embed_version", EMBED_VERSION
                            ]
                            
                            # 添加特定参数
                            for k, v in spec_dict.items():
                                cmd.extend([k, str(v)])
                            
                            run_cmd(cmd)

if __name__ == "__main__":
    main()

