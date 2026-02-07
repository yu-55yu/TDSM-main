import numpy as np
import os
import pickle

# --- 1. 路径与配置 ---
RAW_DIR = "/root/autodl-tmp"                      # 原始大文件目录
FEATS_DIR = "./data/sk_feats"                     # 特征存放目录
LABEL_DIR = "./data/label_splits/ntu60"           # 协议文件目录

def load_protocol_ids(split_name):
    """直接读取模型使用的协议文件，确保 ID 完全对齐"""
    path = os.path.join(LABEL_DIR, f"ru{split_name}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到协议文件: {path}")
    ids = np.load(path)
    print(f">>> 已加载 ru{split_name}.npy，包含 {len(ids)} 个未知类")
    return ids.tolist()

def load_pkl_labels(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return np.array(data[1])

def process_partition(split_name):
    print(f"\n--- 正在生成 shift_ntu60_{split_name}_r 划分 (恢复 ztest 命名) ---")
    
    # 动态加载 ru5.npy 或 ru12.npy
    unseen_ids = load_protocol_ids(split_name)
    save_path = os.path.join(FEATS_DIR, f"shift_ntu60_{split_name}_r")
    os.makedirs(save_path, exist_ok=True)

    # 内存优化加载
    train_feat = np.load(os.path.join(RAW_DIR, 'train_25joint.npy'), mmap_mode='r')
    test_feat = np.load(os.path.join(RAW_DIR, 'test_25joint.npy'), mmap_mode='r')
    train_label = load_pkl_labels(os.path.join(RAW_DIR, 'train_label.pkl'))
    test_label = load_pkl_labels(os.path.join(RAW_DIR, 'val_label.pkl'))

    # A. 训练集 (Seen Only)
    seen_mask = ~np.isin(train_label, unseen_ids)
    np.save(os.path.join(save_path, 'train.npy'), train_feat[seen_mask])
    np.save(os.path.join(save_path, 'train_label.npy'), train_label[seen_mask])
    print(f"   - Train (Seen) 已保存: {train_label[seen_mask].shape[0]} 样本")

    # B. Z-Test (Unseen Only) - 恢复原名 ztest
    unseen_mask = np.isin(test_label, unseen_ids)
    np.save(os.path.join(save_path, 'ztest.npy'), test_feat[unseen_mask])
    np.save(os.path.join(save_path, 'z_label.npy'), test_label[unseen_mask])
    print(f"   - Z-Test (Unseen) 已保存: {test_label[unseen_mask].shape[0]} 样本")

    # C. G-Test (Full)
    np.save(os.path.join(save_path, 'gtest.npy'), test_feat)
    np.save(os.path.join(save_path, 'g_label.npy'), test_label)
    print(f"   - G-Test (Full) 已完成")

    del train_feat, test_feat

if __name__ == "__main__":
    process_partition("5")
    process_partition("12")
    print("\n✅ 数据划分已全部完成。")