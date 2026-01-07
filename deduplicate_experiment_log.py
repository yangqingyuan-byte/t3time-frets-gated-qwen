#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验日志去重脚本
对于完全重复的实验日志内容，只保留一个
"""
import json
import os
import sys
import shutil
from datetime import datetime
from collections import OrderedDict

def normalize_json(obj):
    """
    规范化JSON对象，确保相同内容的JSON字符串一致
    通过重新序列化并排序键来实现
    """
    # 使用sort_keys=True确保相同内容的JSON字符串一致
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)

def deduplicate_log(input_file, output_file=None, backup=True, dry_run=False):
    """
    对实验日志进行去重
    
    Args:
        input_file: 输入日志文件路径
        output_file: 输出文件路径（如果为None，则覆盖原文件）
        backup: 是否备份原文件
        dry_run: 是否为试运行模式（不实际修改文件）
    
    Returns:
        (原始数量, 去重后数量, 重复数量)
    """
    if not os.path.exists(input_file):
        print(f"❌ 错误: 找不到日志文件 {input_file}")
        return None, None, None
    
    print("="*80)
    print("实验日志去重工具")
    print("="*80)
    print(f"输入文件: {input_file}")
    
    if output_file is None:
        output_file = input_file
        print(f"输出文件: {output_file} (覆盖原文件)")
    else:
        print(f"输出文件: {output_file}")
    
    if dry_run:
        print("⚠️  试运行模式: 不会实际修改文件")
    print("="*80)
    
    # 读取所有日志条目
    print("\n📖 正在读取日志文件...")
    seen_records = OrderedDict()  # 使用OrderedDict保持顺序
    duplicate_count = 0
    line_number = 0
    duplicate_examples = []  # 保存前几个重复的例子
    max_examples = 5  # 最多显示5个重复例子
    json_errors = []  # 保存JSON解析错误
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            line_number += 1
            
            try:
                # 解析JSON
                data = json.loads(line)
                
                # 规范化JSON字符串作为去重key
                normalized = normalize_json(data)
                
                # 检查是否已存在
                if normalized not in seen_records:
                    seen_records[normalized] = (line_num, line, data)
                else:
                    duplicate_count += 1
                    # 保存前几个重复例子
                    existing_line_num, existing_line, existing_data = seen_records[normalized]
                    if len(duplicate_examples) < max_examples:
                        duplicate_examples.append((line_num, existing_line_num))
                    
            except json.JSONDecodeError as e:
                # 对于无法解析的行，也保留
                if line not in seen_records:
                    seen_records[line] = (line_num, line, None)
                else:
                    duplicate_count += 1
                    if len(json_errors) < 3:
                        json_errors.append((line_num, str(e)))
    
    # 显示处理进度
    if line_number > 0:
        print(f"  ✅ 已处理 {line_number} 行")
    
    original_count = line_number
    unique_count = len(seen_records)
    
    print(f"\n📊 统计信息:")
    print(f"  原始日志条目数: {original_count}")
    print(f"  去重后条目数:   {unique_count}")
    print(f"  重复条目数:     {duplicate_count}")
    if original_count > 0:
        print(f"  去重率:         {duplicate_count/original_count*100:.2f}%")
    
    # 显示重复例子
    if duplicate_examples:
        print(f"\n📋 重复示例（前{len(duplicate_examples)}个）:")
        for dup_line, orig_line in duplicate_examples:
            print(f"  行 {dup_line} 与行 {orig_line} 完全相同")
    
    # 显示JSON解析错误
    if json_errors:
        print(f"\n⚠️  JSON解析错误（前{len(json_errors)}个）:")
        for line_num, error in json_errors:
            print(f"  行 {line_num}: {error}")
    
    if duplicate_count == 0:
        print("\n✅ 未发现重复条目，无需去重")
        return original_count, unique_count, duplicate_count
    
    if dry_run:
        print("\n⚠️  试运行模式: 未实际修改文件")
        return original_count, unique_count, duplicate_count
    
    # 备份原文件
    if backup and output_file == input_file:
        backup_file = f"{input_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n💾 正在备份原文件到: {backup_file}")
        shutil.copy2(input_file, backup_file)
        print(f"✅ 备份完成")
    
    # 写入去重后的结果
    print(f"\n💾 正在写入去重后的结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for normalized_key, (line_num, line, data) in seen_records.items():
            f.write(line + '\n')
    
    print(f"✅ 去重完成！")
    print(f"\n📁 文件信息:")
    print(f"  原始文件大小: {os.path.getsize(input_file if backup else input_file) / 1024 / 1024:.2f} MB")
    if os.path.exists(output_file):
        print(f"  输出文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    return original_count, unique_count, duplicate_count

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='实验日志去重工具 - 对完全重复的实验日志内容，只保留一个',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 试运行模式（不实际修改文件）
  python deduplicate_experiment_log.py --dry-run
  
  # 去重并覆盖原文件（自动备份）
  python deduplicate_experiment_log.py
  
  # 去重并保存到新文件
  python deduplicate_experiment_log.py --output experiment_results_dedup.log
  
  # 去重但不备份
  python deduplicate_experiment_log.py --no-backup
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='experiment_results.log',
        help='输入日志文件路径（默认: experiment_results.log）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出文件路径（默认: 覆盖原文件）'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='不备份原文件（默认会备份）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='试运行模式：只显示统计信息，不实际修改文件'
    )
    
    args = parser.parse_args()
    
    # 执行去重
    original_count, unique_count, duplicate_count = deduplicate_log(
        input_file=args.input,
        output_file=args.output,
        backup=not args.no_backup,
        dry_run=args.dry_run
    )
    
    if original_count is None:
        sys.exit(1)
    
    print("\n" + "="*80)
    print("去重完成！")
    print("="*80)
    
    if duplicate_count > 0 and not args.dry_run:
        print(f"\n💡 提示: 原文件已备份，如需恢复可使用备份文件")

if __name__ == "__main__":
    main()
