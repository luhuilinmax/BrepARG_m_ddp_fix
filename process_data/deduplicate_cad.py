import os
import pickle
import argparse
import random
from tqdm import tqdm
from hashlib import sha256
from convert_utils import *


def resolve_output_path(args):
    if args.output:
        return args.output
    if args.option == "deepcad":
        return f"data/deepcad_data_split_{args.bit}bit.pkl"
    if args.option == "abc":
        return f"data/abc_data_split_{args.bit}bit.pkl"
    return f"data/furniture_data_split_{args.bit}bit.pkl"


def load_input_paths(input_path):
    """
    Load pkl file paths from a directory or from various pkl formats.
    Supports:
    - Directory: recursively find all .pkl files
    - Split pkl (dict with 'train' key): return the train list
    - List pkl: return the list
    - Single sample pkl (dict with 'surf_wcs'): return [path]
    """
    if os.path.isdir(input_path):
        pkl_paths = []
        for root, _, files in os.walk(input_path):
            for name in files:
                if name.endswith(".pkl"):
                    pkl_paths.append(os.path.join(root, name))
        pkl_paths = sorted(pkl_paths)
        print(f"检测到目录输入，递归找到 {len(pkl_paths)} 个 pkl 文件")
        return pkl_paths

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "train" in data and isinstance(data["train"], list):
        print(f"检测到 split pkl，使用 train 列表，共 {len(data['train'])} 条")
        return data["train"]

    if isinstance(data, list):
        print(f"检测到路径列表 pkl，共 {len(data)} 条")
        return data

    if isinstance(data, dict) and "surf_wcs" in data:
        print("检测到单样本 pkl，按单文件处理")
        return [input_path]

    raise ValueError(
        "无法识别输入 pkl 结构。请提供单样本 pkl、包含 train 的 split pkl，或路径列表 pkl。"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/abc_parsed",
        help="Path to the data folder (directory containing .pkl files, or a pkl file)",
    )
    parser.add_argument(
        "--input_pkl",
        type=str,
        default=None,
        help="Input pkl path (alternative to --data, supports directory, split pkl, list pkl or single sample pkl)",
    )
    parser.add_argument("--bit", type=int, default=6, help="Deduplication precision (bit)")
    parser.add_argument(
        "--option",
        type=str,
        choices=["abc", "deepcad", "furniture"],
        default="abc",
        help="Select dataset type [abc/deepcad/furniture] (default: abc)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output split pkl path")
    args = parser.parse_args()

    # Choose input source: prefer --input_pkl if provided, otherwise --data
    input_source = args.input_pkl if args.input_pkl is not None else args.data
    all_pkl_paths = load_input_paths(input_source)

    # Split dataset: 90% train, 5% val, 5% test (original logic)
    random.seed(42)
    random.shuffle(all_pkl_paths)

    total_count = len(all_pkl_paths)
    train_count = int(total_count * 0.9)
    val_count = int(total_count * 0.95)

    train_all = all_pkl_paths[:train_count]
    val_path = all_pkl_paths[train_count:val_count]
    test_path = all_pkl_paths[val_count:]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_all)} files")
    print(f"  Val:   {len(val_path)} files")
    print(f"  Test:  {len(test_path)} files")

    # Deduplicate training set only
    print(f"\nStart deduplicating training set...")
    train_path = []
    unique_hash = set()
    total = 0

    for path_idx, pkl_path in tqdm(
        enumerate(train_all), total=len(train_all), desc="Deduplicating train set"
    ):
        total += 1
        try:
            with open(pkl_path, "rb") as file:
                data = pickle.load(file)
        except Exception as e:
            print(f"Failed to read {pkl_path}: {e}")
            continue

        if "surf_wcs" not in data:
            print(f"Missing key 'surf_wcs' in {pkl_path}, skipped")
            continue

        surfs_wcs = data["surf_wcs"]
        surf_hash_total = []
        for surf in surfs_wcs:
            np_bit = real2bit(surf, n_bits=args.bit).reshape(-1, 3)
            surf_hash_total.append(sha256(np_bit.tobytes()).hexdigest())

        data_hash = "_".join(sorted(surf_hash_total))
        prev_len = len(unique_hash)
        unique_hash.add(data_hash)
        if prev_len < len(unique_hash):
            train_path.append(pkl_path)

        if path_idx % 2000 == 0:
            print(f"Deduplication rate: {len(unique_hash) / total:.2%}")

    output_path = resolve_output_path(args)

    print(f"\nTraining set deduplication finished:")
    print(f"  Before: {len(train_all)} files")
    print(f"  After: {len(train_path)} files")
    print(f"  Duplicates removed: {len(train_all) - len(train_path)} files")
    if len(train_all) > 0:
        print(f"  Retention rate: {len(train_path) / len(train_all):.2%}")

    data_path = {
        "train": train_path,
        "val": val_path,
        "test": test_path,
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as tf:
        pickle.dump(data_path, tf)

    print(f"\nFinal dataset statistics:")
    print(f"  Train: {len(train_path)} files (deduplicated)")
    print(f"  Val:   {len(val_path)} files")
    print(f"  Test:  {len(test_path)} files")
    print(f"  Total: {len(train_path) + len(val_path) + len(test_path)} files")
    print(f"\nResult saved to: {output_path}")


if __name__ == "__main__":
    main()
