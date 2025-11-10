# EAT Algorithm Implementation


This repo is an implementation of our paper "**EAT: QoS-Aware Edge-Collaborative AIGC Task Scheduling via Attention-Guided Diffusion Reinforcement Learning**", submitted to IEEE Transactions on Mobile Computing (TMC).

We propose EAT, a QoS-aware Edge-collaborative AIGC Task Scheduling algorithm. In this repo, we deploy the EAT algorithm, along with baseline methods, and implement Stable Diffusion on server nodes for AIGC task scheduling and evaluation.

<div align=center>
  <img src="https://github.com/user-attachments/assets/2106305e-5c23-4486-9436-4a75be484fa1" width="1024px">
</div>

## 项目结构速览

- `EAT Simu/`：离线仿真与训练环境，包含 `train_*.py` 训练脚本、`evaluate_*.py` 基线评估脚本、强化学习策略实现以及扩散模型组件。
- `EAT Real/`：真实部署实验环境，提供与仿真环境一致的调度策略实现，以及对真实边缘节点的任务下发与评估脚本。
- `Server/`：边缘/服务器端 Stable Diffusion 部署脚本，包括任务描述模板（`template.py`）、任务通信工具（`tools.py`）、分布式推理主循环（`run_sd.py`）以及节点监听入口（`run_node.py`）。
- `Server/distrifuser/`：Stable Diffusion 分布式推理依赖模块。
- `Server/scripts/`：模型性能评估、COCO 数据处理与示例脚本。
- `README.md`：使用指南与论文简介。

> 说明：`Server` 目录下的核心脚本已补充中文逐行注释，便于理解任务调度与分布式推理流程。

## EAT Simu 环境与执行

### 环境依赖

- Python 3.10+（推荐使用 Conda 或 venv）
- 依赖列表见 `EAT Simu/requirements.txt`，可直接安装：

  ```bash
  pip install -r "EAT Simu/requirements.txt"
  ```
  
- 若需要 GPU 训练，请额外根据显卡安装匹配的 CUDA 版 PyTorch。

### 运行流程

1. 进入仿真目录并选择训练脚本：

   ```bash
   cd "EAT Simu"
   python train_adsac.py
   ```

   训练脚本会读取 `_SDEnv/` 提供的仿真环境，利用扩散近似模型评估调度策略表现。

2. 训练完成后，可使用评估脚本比较不同策略：

   ```bash
   python evaluate_adsac.py  # 评估 ADSAC 策略
   python evaluate_random.py # 对比随机策略
   ```

3. 训练得到的策略会保存在 `EAT Simu/res_policy/` 中，可进一步转换或上传至真实部署环境使用。

> EAT Simu 全流程均在单机仿真环境内执行，不会与 `Server/` 目录中的真实 Stable Diffusion 节点发生联动。

## EAT Real 与 Server 环境与执行

### 环境依赖

- Python 3.10+，建议与执行节点共用相同虚拟环境。
- 依赖列表见 `EAT Real/requirements.txt`，建议在调度端与所有 Server 节点安装：

  ```bash
  pip install -r "EAT Real/requirements.txt"
  ```

- 已下载的 Stable Diffusion 权重（默认 `CompVis/stable-diffusion-v1-4`，需提前缓存或配置镜像）。
- 节点间网络互通，并开放调度端到各工作节点的端口（默认 16122）。

### 节点部署（Server 侧）

1. 为每台执行节点部署本仓库 `Server/` 目录，确保权重文件可访问。
2. 在节点上启动监听脚本：

   ```bash
   cd Server
   python run_node.py
   ```

   脚本会在端口 `16122` 监听，收到调度端指令后通过 `torchrun` 拉起 `run_sd.py` 执行 Stable Diffusion 推理。

### 调度执行（EAT Real 侧）

1. 将在仿真阶段训练得到的策略文件拷贝至 `EAT Real/policy/` 或 `EAT Real/upload_policy/`。
2. 根据评估需求运行对应脚本（会实际向 Server 端节点下发任务）：

   ```bash
   cd "EAT Real"
   python evaluate_adsac.py
   ```

3. `EAT Real` 会根据策略生成 `StableDiffusionCommand`，通过网络发送给各个执行节点；`Server/run_sd.py` 完成图像生成并回传结果（或状态反馈）。

> 真实部署中，`EAT Real` 与 `Server/` 目录形成调度端与执行端的配合：前者负责决策与指令下发，后者负责实机推理。

## 进一步阅读

- 论文草稿与更多实验细节将在论文正式公开后同步更新。
- 若需了解代码实现细节，请参考各脚本中新增的中文注释与函数参数说明。
