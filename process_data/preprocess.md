python -B process_brep.py --input /root/autodl-fs/00710004_32879ef7e9e47ed44bd9e64f_step_000.step --output /root/autodl-fs/preprocessed_data/abc_0071_step_v00_qian100_breparg


已经按你确认的方向改好了，两个文件都已更新：

autodl-tmp/AR/BrepARG_m/process_data/deduplicate_cad.py

改为支持 --input_pkl。
输入可兼容三种：
单样本 pkl（你这个 process_brep.py 产物）
旧的 split pkl（含 train）
路径列表 pkl（list[str]）
输出只保留 train 和 val，并且 val == train。
不再生成 test。
autodl-tmp/AR/BrepARG_m/process_data/deduplicate_se_data.py

--data_list 现在同样兼容三种输入（单样本 / split / 路径列表）。
单样本 pkl 可以直接跑，不再依赖必须有 train/test 结构。
保存输出时处理了无目录路径的情况，避免 os.makedirs('') 报错。
另外我已经做过语法检查，两份脚本都通过。

你可以直接用下面命令：

# 1) CAD 去重（输入你的单样本 pkl，输出 train/val 一致的 split pkl）
python /root/autodl-tmp/AR/BrepARG_m/process_data/deduplicate_cad.py \
  --input_pkl /root/autodl-fs/preprocessed_data/abc_0071_step_v00_qian100_breparg/00710004_32879ef7e9e47ed44bd9e64f_step_000.pkl \
  --bit 6 \
  --option abc \
  --output /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_single_data_split_6bit.pkl
# 2) SE 去重（face）
python /root/autodl-tmp/AR/BrepARG_m/process_data/deduplicate_se_data.py \
  --data_list /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_single_data_split_6bit.pkl \
  --mode face \
  --bit 6 \
  --output /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_single_unique_surfaces.pkl

# 3) SE 去重（edge）
python /root/autodl-tmp/AR/BrepARG_m/process_data/deduplicate_se_data.py \
  --data_list /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_single_data_split_6bit.pkl \
  --mode edge \
  --bit 6 \
  --output /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_single_unique_edges.pkl



# 三个文件分别做什么预处理？

process_brep.py（STEP→pkl） → deduplicate_cad.py（划分 + 整模去重训练集） → 训练主任务用该 split；若要做表面/边 token 的去重统计，再在 train 列表 上跑 deduplicate_se_data.py

## 1. process_brep.py：STEP → 结构化 B-rep 特征（单样本 .pkl）
输入：--input 下的 STEP（通过 load_step 得到目录/文件列表），--output 为输出目录。  
单文件流程（多进程 Pool）：  
只保留 单个 solid 的模型（len(cad_solid)!=1 则丢弃）。  
split_all_closed_faces / split_all_closed_edges（num_splits=0，语义上依赖 occwl 的封闭面/边处理）。  
面数上限：len(faces) > MAX_FACE（200）则丢弃，控制规模。  
extract_primitive(solid)：从 solid 抽出面采样点、边采样点、边端点、面–边、边–面等关联。  
normalize：  
全体点做 全局 平移+缩放，映射到约 [-1, 1]（WCS：surfs_wcs / edges_wcs / corner_wcs）。  
每个面、每条边再各自做 局部 归一化，得到 NCS：surf_ncs / edge_ncs。  
角点：corner_wcs 四舍五入后按坐标去重，得到 corner_unique，并建 edgeCorner_IncM。  
在 WCS 下算每个面、每条边的 轴对齐包围盒 surf_bbox_wcs / edge_bbox_wcs。  
存成 float32 字典并 pickle 到 output/<step文件名>.pkl。  
概括：几何解析 + 全局/局部归一化 + 拓扑邻接 + 包围盒 + 角点合并，是后面所有列表/去重脚本的原始样本格式。

## 2. deduplicate_cad.py：整模型级 划分与训练集去重
输入：某数据集根目录下全部解析好的 .pkl。  
输出：data/{abc|deepcad|furniture}_data_split_{bit}bit.pkl。  
作用：  
按上面规则做 90/5/5 的 train/val/test 路径列表；  
仅在 train 上按 整模型 的 surf_wcs 量化指纹去重，避免训练集里几何重复；  
val/test 保持划分不变、不去重。

## 3. deduplicate_se_data.py：面或边 NCS 片段级 去重（为 VQ 等准备「字典」式数据）
输入：已有划分文件 --data_list（默认 data/abc_data_split_6bit.pkl），只读其中的 train 路径。  
--mode：  
face：对每条样本的 surf_ncs 中 每个面 分别处理；  
edge：对 edge_ncs 中 每条边 分别处理。  
做法：把每个面/边的 NCS 点云用 real2bit（默认 6 bit）量化，sha256 作为 key；全局 unique_hash 没见过的才把 原始 float 数组 append 到 unique_data。  
输出：默认 data/{dataset_type}_parsed_unique_surfaces.pkl 或 ..._unique_edges.pkl（也可用 --output 指定）。  
概括：从训练集里收集 不重复的局部几何 token（面片或边曲线在 NCS 下的表示），用于后续例如 surface/edge VQ-VAE 码本统计或训练，不是再划分 train/val/test。  

