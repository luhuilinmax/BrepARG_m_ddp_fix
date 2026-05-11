import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import Pool
from convert_utils import *
from occwl.io import load_step as load_solids_from_step
import shutup; shutup.please()


# To speed up processing, define maximum threshold
MAX_FACE = 200


def dump_data_dict_inspection(data, report_path, max_preview_elements=128):
    """
    将组装的 data 字典写成文本：每个 key 的名称、类型、维度/长度、dtype，
    以及数值预览（小数组全文，大数组统计量 + 前若干元素）。
    """
    lines = []

    def append_array(name, arr):
        lines.append(f"  shape: {arr.shape}")
        lines.append(f"  dtype: {arr.dtype}")
        flat = np.ravel(arr)
        n = flat.size
        if n == 0:
            lines.append("  values: (empty)")
            return
        if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
            fi = flat.astype(np.float64) if np.issubdtype(arr.dtype, np.floating) else flat.astype(np.int64)
            lines.append(f"  min: {fi.min()}, max: {fi.max()}")
            if np.issubdtype(arr.dtype, np.floating):
                lines.append(f"  mean: {float(fi.mean())}")
        if n <= max_preview_elements:
            lines.append(f"  values (full):\n{np.array2string(arr, threshold=np.inf, max_line_width=120)}")
        else:
            prev = flat[:max_preview_elements]
            lines.append(
                f"  values (preview, first {max_preview_elements} elements raveled): {prev}"
            )

    for key, val in data.items():
        lines.append("=" * 72)
        lines.append(f"key: {key}")
        lines.append(f"type: {type(val).__name__}")
        if isinstance(val, np.ndarray):
            append_array(key, val)
        elif isinstance(val, (list, tuple)):
            lines.append(f"  len: {len(val)}")
            if len(val) == 0:
                lines.append("  values: (empty sequence)")
            else:
                first = val[0]
                lines.append(f"  elem[0] type: {type(first).__name__}")
                if len(val) <= 32:
                    lines.append(f"  values (full): {val}")
                else:
                    lines.append(f"  values (first 32): {list(val[:32])} ...")
        else:
            s = repr(val)
            lines.append(f"  repr: {s if len(s) <= 2000 else s[:2000] + '...<truncated>'}")
        lines.append("")

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[inspect] wrote {report_path}")


def normalize(surf_pnts, edge_pnts, corner_pnts):
    """
    Various levels of normalization 
    """
    # Global normalization to -1~1
    total_points = np.array(surf_pnts).reshape(-1, 3)
    min_vals = np.min(total_points, axis=0)
    max_vals = np.max(total_points, axis=0)
    global_offset = min_vals + (max_vals - min_vals)/2 
    global_scale = max(max_vals - min_vals)
    assert global_scale != 0, 'scale is zero'

    surfs_wcs, edges_wcs, surfs_ncs, edges_ncs = [],[],[],[]

    # Normalize corner 
    corner_wcs = (corner_pnts - global_offset[np.newaxis,:]) / (global_scale * 0.5)

    # Normalize surface
    for surf_pnt in surf_pnts:    
        # Normalize CAD to WCS
        surf_pnt_wcs = (surf_pnt - global_offset[np.newaxis,np.newaxis,:]) / (global_scale * 0.5)
        surfs_wcs.append(surf_pnt_wcs)
        # Normalize Surface to NCS
        min_vals = np.min(surf_pnt_wcs.reshape(-1,3), axis=0)
        max_vals = np.max(surf_pnt_wcs.reshape(-1,3), axis=0)
        local_offset = min_vals + (max_vals - min_vals)/2 
        local_scale = max(max_vals - min_vals)
        pnt_ncs = (surf_pnt_wcs - local_offset[np.newaxis,np.newaxis,:]) / (local_scale * 0.5)
        surfs_ncs.append(pnt_ncs)
       
    # Normalize edge
    for edge_pnt in edge_pnts:    
        # Normalize CAD to WCS
        edge_pnt_wcs = (edge_pnt - global_offset[np.newaxis,:]) / (global_scale * 0.5)
        edges_wcs.append(edge_pnt_wcs)
        # Normalize Edge to NCS
        min_vals = np.min(edge_pnt_wcs.reshape(-1,3), axis=0)
        max_vals = np.max(edge_pnt_wcs.reshape(-1,3), axis=0)
        local_offset = min_vals + (max_vals - min_vals)/2 
        local_scale = max(max_vals - min_vals)
        pnt_ncs = (edge_pnt_wcs - local_offset) / (local_scale * 0.5)
        edges_ncs.append(pnt_ncs)
        assert local_scale != 0, 'scale is zero'

    surfs_wcs = np.stack(surfs_wcs)
    surfs_ncs = np.stack(surfs_ncs)
    edges_wcs = np.stack(edges_wcs)
    edges_ncs = np.stack(edges_ncs)

    return surfs_wcs, edges_wcs, surfs_ncs, edges_ncs, corner_wcs


def parse_solid(solid):
    """
    Parse the surface, curve, face, edge, vertex in a CAD solid.
   
    Args:
    - solid (occwl.solid): A single brep solid in occwl data format.

    Returns:
    - data: A dictionary containing all parsed data
    """
    assert isinstance(solid, Solid)

    # Split closed surface and closed curve to halve
    solid = solid.split_all_closed_faces(num_splits=0)
    solid = solid.split_all_closed_edges(num_splits=0)

    if len(list(solid.faces())) > MAX_FACE:
        return None

    face_pnts, edge_pnts, edge_corner_pnts, edgeFace_IncM, faceEdge_IncM = extract_primitive(solid)

    surfs_wcs, edges_wcs, surfs_ncs, edges_ncs, corner_wcs = normalize(face_pnts, edge_pnts, edge_corner_pnts)

    # Remove duplicate and merge corners 
    corner_wcs = np.round(corner_wcs, 4) 
    corner_unique = []
    for corner_pnt in corner_wcs.reshape(-1, 3):
        if len(corner_unique) == 0:
            corner_unique = corner_pnt.reshape(1, 3)
        else:
            exists = np.any(np.all(corner_unique == corner_pnt, axis=1))
            if exists:
                continue 
            else:
                corner_unique = np.concatenate([corner_unique, corner_pnt.reshape(1, 3)], 0)

    # Edge-corner adjacency  
    edgeCorner_IncM = []
    for edge_corner in corner_wcs:
        start_corner_idx = np.where((corner_unique == edge_corner[0]).all(axis=1))[0].item()
        end_corner_idx = np.where((corner_unique == edge_corner[1]).all(axis=1))[0].item()
        edgeCorner_IncM.append([start_corner_idx, end_corner_idx])
    edgeCorner_IncM = np.array(edgeCorner_IncM)

    # Surface global bbox
    surf_bboxes = []
    for pnts in surfs_wcs:
        min_point, max_point = get_bbox(pnts.reshape(-1, 3))
        surf_bboxes.append(np.concatenate([min_point, max_point]))
    surf_bboxes = np.vstack(surf_bboxes)

    # Edge global bbox
    edge_bboxes = []
    for pnts in edges_wcs:
        min_point, max_point = get_bbox(pnts.reshape(-1, 3))
        edge_bboxes.append(np.concatenate([min_point, max_point]))
    edge_bboxes = np.vstack(edge_bboxes)

    # Convert to float32 to save space
    data = {
        'surf_wcs': surfs_wcs.astype(np.float32),
        'edge_wcs': edges_wcs.astype(np.float32),
        'surf_ncs': surfs_ncs.astype(np.float32),
        'edge_ncs': edges_ncs.astype(np.float32),
        'corner_wcs': corner_wcs.astype(np.float32),
        'edgeFace_adj': edgeFace_IncM,
        'edgeCorner_adj': edgeCorner_IncM,
        'faceEdge_adj': faceEdge_IncM,
        'surf_bbox_wcs': surf_bboxes.astype(np.float32),
        'edge_bbox_wcs': edge_bboxes.astype(np.float32),
        'corner_unique': corner_unique.astype(np.float32),
    }

    return data


def process(args):
    step_folder, OUTPUT, INPUT_ROOT, inspect = args
    try:
        # Load cad data
        if step_folder.endswith('.step'):
            step_path = step_folder 
        else:
            for _, _, files in os.walk(step_folder):
                assert len(files) == 1 
                step_path = os.path.join(step_folder, files[0])

        cad_solid = load_solids_from_step(step_path)
        if len(cad_solid) != 1: 
            return 0 
        data = parse_solid(cad_solid[0])
        if data is None: 
            return 0

        # Save directly under the OUTPUT folder
        base_name = os.path.splitext(os.path.basename(step_path))[0]
        os.makedirs(OUTPUT, exist_ok=True)
        save_path = os.path.join(OUTPUT, base_name + '.pkl')
        with open(save_path, "wb") as tf:
            pickle.dump(data, tf)
        if inspect:
            inspect_path = os.path.join(OUTPUT, base_name + '_data_inspect.txt')
            dump_data_dict_inspection(data, inspect_path)
        return 1 
    except Exception as e:
        return 0


def load_step_by_range(root_dir, start_idx, end_idx):
    """
    从 root_dir 下按子文件夹名称（零填充数字）筛选编号在 [start_idx, end_idx] 内的
    子文件夹，并收集其中所有 .step 文件路径（按文件夹编号升序）。

    Args:
    - root_dir  (str): 数据根目录，如 /data/dataset/ABC/uz
    - start_idx (int): 起始编号（含），对应文件夹名如 00007643
    - end_idx   (int): 结束编号（含）

    Returns:
    - step_files (list[str]): 筛选后的 .step 文件路径列表
    """
    try:
        all_entries = sorted(os.listdir(root_dir))
    except FileNotFoundError:
        raise FileNotFoundError(f"输入目录不存在: {root_dir}")

    step_files = []
    for entry in all_entries:
        entry_path = os.path.join(root_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        try:
            folder_idx = int(entry)
        except ValueError:
            continue
        if folder_idx < start_idx or folder_idx > end_idx:
            continue
        for dirpath, _, files in os.walk(entry_path):
            for filename in files:
                if filename.lower().endswith('.step'):
                    step_files.append(os.path.join(dirpath, filename))

    return step_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='处理 ABC 数据集中指定编号范围的 B-rep 文件'
    )
    parser.add_argument(
        "--input", type=str,
        default='/workspace/dataset/uz',
        help="数据根目录，子文件夹以零填充数字命名（如 00007643）"
    )
    parser.add_argument(
        "--output", type=str,
        default='/workspace/data/brep_parsed',
        help="输出目录"
    )
    parser.add_argument(
        "--start", type=int, required=True,
        help="起始编号（含），例如 7643 对应文件夹 00007643"
    )
    parser.add_argument(
        "--end", type=int, required=True,
        help="结束编号（含），例如 8000 对应文件夹 00008000"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="并行进程数，默认 1（单进程调试友好）；0 表示自动使用全部 CPU 核心"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="同时输出 <stem>_data_inspect.txt（keys、shapes、dtypes、统计量、数值预览）"
    )
    args = parser.parse_args()

    if args.start > args.end:
        raise ValueError(f"--start ({args.start}) 不能大于 --end ({args.end})")

    print(f"[INFO] 输入目录 : {args.input}")
    print(f"[INFO] 输出目录 : {args.output}")
    print(f"[INFO] 处理范围 : [{args.start}, {args.end}]（含两端）")

    step_files = load_step_by_range(args.input, args.start, args.end)

    if len(step_files) == 0:
        print("[WARNING] 在指定范围内未找到任何 .step 文件，请检查路径和范围。")
        exit(0)

    print(f"[INFO] 共找到 {len(step_files)} 个 .step 文件，开始处理...")

    num_workers = args.workers if args.workers != 0 else os.cpu_count()

    if num_workers == 1:
        # 单进程（便于调试）
        valid = 0
        for step_file in tqdm(step_files, total=len(step_files)):
            status = process((step_file, args.output, args.input, args.inspect))
            valid += status
    else:
        # 多进程并行
        valid = 0
        convert_iter = Pool(num_workers).imap(
            process,
            [(sf, args.output, args.input, args.inspect) for sf in step_files]
        )
        for status in tqdm(convert_iter, total=len(step_files)):
            valid += status

    print(f'Done... Data Converted Ratio {100.0 * valid / len(step_files):.2f}%')
