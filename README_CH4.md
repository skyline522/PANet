
# Chapter 4 Code — TCC-Head (Topology/Connectivity Consistency Head)

本目录在第三章（论文一致版本：SPMM + SCPM + CSRH + EPML）基础上，新增**第四章焊缝优化**：

## 1) TCC-Head（必选，本目录默认启用）
- 在查询门控特征 `y_Q` 上新增结构分支：预测 `skeleton`（中心线）与 `edge`（边界）
- 训练时由 Query GT 自动生成 `skel_gt` 与 `edge_gt`，并加入结构监督
- 同时将 `skel/edge` 概率图轻量融合到前景相似响应（不改变第三章主干）

对应实现：
- 模型：`dpanet_pml.py` 中 `TCCHead` + `DPANet_PML` 的 fusion 逻辑
- Target & loss：`util/tcc_ops.py`
- 训练集成：`train.py`（tcc_* 超参在 `config.py`）

## 2) LDR（可选，默认关闭）
- 仅当你想“提出跨产线鲁棒性提升方法”时启用
- 不启用也完全可以做跨产线实验（那属于“难度评测”而不是“方法增强”）
- 启用方式：`config.py` 中 `use_ldr=True` 且 `loss.ldr_weight>0`

实现：
- `util/ldr_aug.py`：纯 torch 的 photometric 域随机化
- `train.py`：teacher-student KL consistency loss

## 3) HANFEG/Weld 数据集
- 新增 loader：`dataloaders/hanfeg.py`
- factory：`dataloaders/customized.py` 里 `hanfeg_fewshot(...)`
- 若你已有更确定的数据目录结构，请按你的实际路径在 `hanfeg.py` 内调整即可。

