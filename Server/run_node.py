import argparse  # 支持命令行参数，方便一机多节点
import tools  # 导入工具函数模块，用于接收任务
import subprocess  # 导入 subprocess 以调用外部命令
from template import StableDiffusionCommand  # 导入命令对象定义

def build_torch_command(cmd: StableDiffusionCommand):
    """
    参数说明：
    cmd: 包含任务信息与分布式配置的 StableDiffusionCommand 实例。
    功能：根据命令对象组装 torchrun 子进程启动所需的参数列表。
    """
    node_num = len(cmd.districonfig.node_ips)  # 统计协作节点数量
    node_id = cmd.districonfig.node_id  # 获取当前节点编号

    torch_command = [  # 构造 torchrun 命令参数
        "torchrun",  # 指定启动 torchrun
        f"--nnodes={node_num}",  # 设置节点总数
        f"--node_rank={node_id}",  # 设置当前节点序号
        "--nproc_per_node=1",  # 每个节点的进程数
        f"--master_addr={cmd.districonfig.master_ip}",  # 主节点地址
        f"--master_port={cmd.districonfig.torch_port}",  # 主节点通信端口
        "run_sd.py",  # 指定执行的脚本
        f"{cmd.to_json()}"  # 将命令对象转为 JSON 传递给子进程
    ]

    return torch_command  # 返回拼装好的命令参数列表

def run_node(command_port):
    """
    参数说明：
    command_port: 当前节点监听任务指令的端口号。
    功能：循环监听任务，构建并执行 torchrun 子进程，直至停止。
    """
    while True:  # 持续运行以处理多个任务
        cmd = tools.receive_task(command_port)  # 从主节点接收任务
        while cmd.districonfig.torch_port == '0':  # 若端口为 0 表示配置未就绪
            cmd = tools.receive_task(command_port)  # 继续等待下一条有效任务
        torch_command = build_torch_command(cmd=cmd)  # 生成 torchrun 命令参数
        print(f"receive torch_command in main:{torch_command}")  # 打印命令以便调试

        result = subprocess.run(torch_command, capture_output=False, text=True, check=True, encoding='utf-8')  # 同步执行 torchrun
        print(f"get sub process output:\n{result.stdout}")  # 输出子进程日志

if __name__ == '__main__':  # 脚本主入口
    parser = argparse.ArgumentParser(description="启动单个 Server 节点监听进程")
    parser.add_argument(
        "--port",
        type=int,
        default=16122,
        help="监听任务指令的端口，默认 16122，可在同机多开时设置为 16123/16124 等",
    )
    args = parser.parse_args()

    # 允许通过 `python run_node.py --port 16123` 的方式一次开多个终端模拟多节点
    run_node(args.port)  # 启动节点监听循环
