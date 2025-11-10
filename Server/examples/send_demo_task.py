"""
发送一个真实的 Stable Diffusion 任务到若干节点，并接收生成的图像。
可在调度脚本运行前，手动验证多节点协同推理与结果回传流程。
"""

import argparse
import os
import socket
import sys
import time
from threading import Thread
from typing import List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_ROOT = os.path.dirname(CURRENT_DIR)
if SERVER_ROOT not in sys.path:
    sys.path.insert(0, SERVER_ROOT)

from template import Task, TaskDistriConfig, StableDiffusionCommand  # noqa: E402


def send_command(ip: str, port: int, command: StableDiffusionCommand, timeout=10):
    """将任务指令发送到指定节点。"""
    payload = command.to_json().encode("utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect((ip, port))
        sock.sendall(payload)


def receive_result(listen_ip: str, listen_port: int, output_path: str, timeout=300):
    """监听节点回传的图片数据并保存，带进度条显示。"""
    from tqdm import tqdm

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((listen_ip, listen_port))
        sock.listen(1)
        sock.settimeout(timeout)
        print(f"[Receiver] 等待结果，监听 {listen_ip}:{listen_port} ...")
        conn, addr = sock.accept()
        with conn:
            print(f"[Receiver] 已连接 {addr} ，开始接收数据")
            with open(output_path, "wb") as f:
                pbar = tqdm(desc="Receiving image", unit="B", unit_scale=True)
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
                pbar.close()
    print(f"[Receiver] 图像已保存：{output_path}")


def dispatch_task(
    node_ips: List[str],
    node_ports: List[int],
    command: StableDiffusionCommand,
):
    """依次向所有节点发送同一个命令（调整 node_id）。"""
    for idx, (ip, port) in enumerate(zip(node_ips, node_ports)):
        command.districonfig.node_id = str(idx)
        print(f"[Dispatcher] 发送到 {ip}:{port} (node_id={idx}) ...", end=" ")
        send_command(ip, port, command)
        print("OK")


def main():
    parser = argparse.ArgumentParser(description="手动下发一个多节点 Stable Diffusion 任务")
    parser.add_argument("--ips", nargs="+", required=True, help="节点 IP 列表")
    parser.add_argument("--ports", nargs="+", required=True, type=int, help="节点端口列表")
    parser.add_argument("--co-num", type=int, default=1, help="协同节点数量（需小于等于节点数）")
    parser.add_argument("--prompt", default="an orange cat, ultra realistic", help="正向提示词")
    parser.add_argument("--negative-prompt", default="low quality, blurry", help="反向提示词")
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
    parser.add_argument("--result-ip", default="0.0.0.0", help="本机监听结果的 IP")
    parser.add_argument("--result-port", type=int, default=26010, help="本机监听结果的端口")
    parser.add_argument("--output", default="demo_result.png", help="输出图片路径")
    parser.add_argument("--master-port", type=int, default=29500, help="torchrun 使用的 master_port")
    args = parser.parse_args()

    if len(args.ips) != len(args.ports):
        raise ValueError("IP 与端口数量必须一致")
    if args.co_num < 1 or args.co_num > len(args.ips):
        raise ValueError("co-num 必须在 [1, 节点数量] 之间")

    selected_ips = args.ips[: args.co_num]
    selected_ports = args.ports[: args.co_num]

    task = Task(
        co_num=args.co_num,
        steps=args.steps,
        info={"prompt": args.prompt, "ng_prompt": args.negative_prompt},
        execute=True,
        valid=True,
        node_ids=list(range(args.co_num)),
    )

    distri_config = TaskDistriConfig(
        node_ips=selected_ips,
        torch_port=str(args.master_port),
        master_ip=selected_ips[0],
        master_res_port=str(args.result_port),
        node_id="0",
    )
    command = StableDiffusionCommand(task=task, districonfig=distri_config)

    receiver = Thread(
        target=receive_result,
        args=(args.result_ip, args.result_port, args.output),
        daemon=True,
    )
    receiver.start()

    start = time.time()
    dispatch_task(selected_ips, selected_ports, command)
    receiver.join()
    print(f"[Main] 任务完成，总耗时 {time.time() - start:.2f} 秒")


if __name__ == "__main__":
    main()
