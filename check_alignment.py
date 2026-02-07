import numpy as np
import json
import torch

# --- 1. 配置路径 (请确保与你的 YAML 一致) ---
LABEL_PATH = './data/label_splits/ntu60/ru5.npy'
JSON_PATH = './data/class_lists/ntu60_parts_llm.json'
VAL_DATA_PATH = './data/sk_feats/shift_ntu60_5_r/g_label.npy'

def audit_data():
    print("--- 🛠️ 开始数据对齐性审计 ---")
    
    # A. 检查协议文件
    unseen_labels = np.load(LABEL_PATH)
    print(f"[1] 协议文件 ru5.npy 中的标签: {unseen_labels}")
    
    # B. 检查 JSON 描述
    with open(JSON_PATH, 'r') as f:
        parts_dict = json.load(f)
    
    # C. 抽样检查对齐关系
    print("\n[2] 抽样验证语义对齐 (请人工核对动作名是否符合常识):")
    sample_ids = unseen_labels[:3] # 取前三个未知类
    for sid in sample_ids:
        action_desc = parts_dict.get(str(sid), {}).get('global', 'NOT FOUND')
        print(f"    - 标签 ID [{sid}] 在 JSON 中的描述为: {action_desc}")
        if action_desc == 'NOT FOUND':
            print(f"      🚨 警告: 标签 {sid} 在描述文件中缺失！")

    # D. 检查测试集标签分布
    try:
        g_labels = np.load(VAL_DATA_PATH)
        unique_val = np.unique(g_labels)
        print(f"\n[3] 你的 g_label.npy 中实际包含的标签: {unique_val}")
        
        # 核心逻辑：检查 val 里的标签是否全在协议内
        intersection = np.intersect1d(unique_val, unseen_labels)
        diff = np.setdiff1d(unique_val, unseen_labels)
        
        if len(diff) > 0:
            print(f"    🚨 严重错误: 测试集包含了非法标签 {diff}，这会导致 KeyError！")
        if len(intersection) == len(unseen_labels):
            print(f"    ✅ 协议一致性通过：测试集标签与 ru5.npy 完全匹配。")
    except Exception as e:
        print(f"    ⚠️ 无法读取测试集标签文件: {e}")

    # E. 检查索引偏移 (0-indexed 陷阱)
    print("\n[4] 索引偏移自检:")
    print(f"    - 如果标签 ID 包含 60，说明是 1-60 计数。")
    print(f"    - 如果标签 ID 包含 0，说明是 0-59 计数。")
    print(f"    - 当前 ru5.npy 范围: {unseen_labels.min()} ~ {unseen_labels.max()}")



if __name__ == "__main__":
    audit_data()