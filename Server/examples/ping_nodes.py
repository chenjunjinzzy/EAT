import argparse
import os
import socket
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_ROOT = os.path.dirname(CURRENT_DIR)
if SERVER_ROOT not in sys.path:
    sys.path.insert(0, SERVER_ROOT)

from template import Task, TaskDistriConfig, StableDiffusionCommand


def build_stop_command(target_ip: str) -> StableDiffusionCommand:
    """
    构造一个 torch_port='0' 的最小指令。节点接收到后不会真正启动 torchrun，
    只用于测试连通性，安全无副作用。
    """
    task = Task(info={"prompt": "ping", "ng_prompt": "none"})
    distri_config = TaskDistriConfig(
        node_ips=[target_ip],
        torch_port="0",
        master_ip=target_ip,
        master_res_port="0",
        node_id="0",
    )
    return StableDiffusionCommand(task=task, districonfig=distri_config)


def send_command(ip: str, port: int, command: StableDiffusionCommand, timeout=5):
    """发送 JSON 指令到指定节点，超时时间默认 5 秒。"""
    payload = command.to_json().encode("utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect((ip, port))
        sock.sendall(payload)


def main():
    parser = argparse.ArgumentParser(description="向节点发送最小指令，快速验证连通性。")
    parser.add_argument(
        "--ips",
        nargs="+",
        required=True,
        help="节点 IP 列表（顺序需与端口一一对应）。",
    )
    parser.add_argument(
        "--ports",
        nargs="+",
        required=True,
        type=int,
        help="节点端口列表。",
    )
    args = parser.parse_args()

    if len(args.ips) != len(args.ports):
        raise ValueError("IP 数量与端口数量必须一致。")

    for ip, port in zip(args.ips, args.ports):
        command = build_stop_command(ip)
        print(f"Sending test command to {ip}:{port} ...", end=" ", flush=True)
        start = time.time()
        try:
            send_command(ip, port, command)
        except Exception as exc:
            print(f"FAILED ({exc})")
        else:
            elapsed = time.time() - start
            print(f"OK ({elapsed:.3f}s)")


if __name__ == "__main__":
    main()
