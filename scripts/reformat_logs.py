import json
import os

LOG_FILE = "experiment_results.log"
BACKUP_FILE = "experiment_results.log.bak"

def reformat_entry(entry):
    # 定义期望的键顺序
    priority_keys = ["data_path", "pred_len", "test_mse", "test_mae", "model"]
    
    new_entry = {}
    
    # 1. 按照优先级添加键
    for key in priority_keys:
        if key in entry:
            new_entry[key] = entry[key]
    
    # 2. 添加 timestamp (通常紧随其后)
    if "timestamp" in entry:
        new_entry["timestamp"] = entry["timestamp"]
        
    # 3. 添加其余所有键
    for key, value in entry.items():
        if key not in new_entry:
            new_entry[key] = value
            
    return new_entry

def main():
    if not os.path.exists(LOG_FILE):
        print(f"找不到日志文件: {LOG_FILE}")
        return

    # 备份原始文件
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"已备份原始日志到: {BACKUP_FILE}")

    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            reformatted = reformat_entry(entry)
            new_lines.append(json.dumps(reformatted, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"跳过无效行: {line[:50]}... 错误: {e}")
            new_lines.append(line + "\n")

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✅ 日志文件已成功重排字段顺序。共处理 {len(new_lines)} 条记录。")

if __name__ == "__main__":
    main()

