## 多节点快速示例

本示例说明如何复用 `run_node.py` 在同一台机器上模拟多个逻辑节点，并向每个节点发送简单指令，用于在正式运行调度器前快速验证连通性。

### 1. 启动多个节点

打开多个终端（或后台运行）分别启动 `run_node.py`，每个节点监听不同的端口：

```bash
# 终端 1
python run_node.py --port 16122

# 终端 2
python run_node.py --port 16123

# 终端 3
python run_node.py --port 16124

# 终端 4
python run_node.py --port 16125
```

若节点部署在不同服务器，只需将命令中的 IP/端口换成真实值，并保持与调度端配置一致即可。

### 2. 发送测试指令

在节点监听就绪后，运行示例客户端。脚本会构造一个最小化的 `StableDiffusionCommand`
（`torch_port="0"`，不会真正启动 `torchrun`），依次发送给参数中的每个节点，用来确认连通性：

```bash
python examples/ping_nodes.py \
  --ips 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 \
  --ports 16122 16123 16124 16125
```

预期输出示例：

```
Sending test command to 127.0.0.1:16122 ...
OK
...
```

若某个节点不可达，脚本会打印连接错误，方便排查防火墙或端口问题。

当所有节点都返回 OK 后，即可在 `EAT Real/evaluate_*.py` 中填入相同的 `node_ips` / `node_ports`
运行完整的调度流程。

### 3. 下发真实任务并接收图像

若想进一步验证推理链路，可使用 `send_demo_task.py` 直接向若干节点分布式下发一个 Stable Diffusion 任务：

```bash
python examples/send_demo_task.py \
  --ips 127.0.0.1 127.0.0.1 \
  --ports 16122 16123 \
  --co-num 2 \
  --prompt "a cute robot painting, watercolor" \
  --negative-prompt "low quality, blurry" \
  --steps 25 \
  --output demo_result.png
```

脚本会在本地开启结果监听端口，等待节点回传生成的 PNG 图片，适合在正式调度前快速验证 “任务→节点推理→结果返回” 的闭环。
