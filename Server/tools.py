import socket  # 导入 socket 用于网络通信
import json  # 导入 json 处理序列化
from template import StableDiffusionCommand  # 导入命令对象定义

def receive_task(command_port)->StableDiffusionCommand:
    """
    参数说明：
    command_port: 监听任务指令的端口号。
    返回值：StableDiffusionCommand 对象或 None（发生异常时）。
    功能：监听套接字，接收来自主节点的任务指令 JSON，并转成命令对象。
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:  # 创建 TCP 套接字并自动释放资源
            sock.bind(('', command_port))  # 绑定到所有可用网卡上的指定端口
            sock.listen()  # 开始监听进入连接
            print(f"Listening for task on port {command_port}...")  # 打印监听信息
            conn, addr = sock.accept()  # 阻塞等待连接请求
            with conn:  # 使用上下文确保连接关闭
                print(f"Connected by {addr}")  # 打印连接来源
                data = b""  # 初始化接收缓冲区
                while True:  # 持续接收数据直到对端关闭
                    packet = conn.recv(4096)  # 每次读取 4096 字节
                    if not packet:  # 为空表示对端关闭
                        break  # 跳出循环
                    data += packet  # 累加数据块
                if data:  # 如果收到了数据
                    cmd_json = data.decode('utf-8')  # 将字节解码为字符串
                    command = StableDiffusionCommand.from_json(cmd_json)  # 反序列化成命令对象
                    print(f"return {command}")  # 打印返回对象
                    return command  # 返回命令
    except Exception as e:
        print(f"Error receiving task: {e}")  # 捕获并打印异常
        return None  # 出错时返回 None

def send_result(ip, port, file_name):
    """
    参数说明：
    ip: 主节点的 IP 地址。
    port: 主节点监听结果的端口。
    file_name: 需要发送的图像文件路径。
    功能：读取本地生成的结果图片并发送到主节点。
    """
    port = int(port)  # 确保端口为整数
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:  # 创建 TCP 连接
            sock.connect((ip, port))  # 主动连接主节点
            print(f"Sending result to {ip}:{port}")  # 打印发送信息
            with open(file_name, 'rb') as f:  # 以二进制方式读取文件
                data = f.read()  # 读取全部文件内容
                sock.sendall(data)  # 将数据一次性发送完毕
    except Exception as e:
        print(f"Error sending result to {ip}:{port}: {e}")  # 打印异常信息

def receive_result(ip, port, output_file):
    """
    参数说明：
    ip: 本地监听的 IP 地址（通常为 0.0.0.0 或本地网卡）。
    port: 本地监听结果的端口。
    output_file: 接收数据后保存的文件路径。
    功能：作为接收端，等待图像数据并写入本地文件。
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:  # 创建 TCP 套接字
            sock.bind((ip, int(port)))  # 绑定到指定 IP 与端口
            sock.listen(1)  # 开始监听，最多排队 1 个连接
            print(f"Listening for incoming data on {ip}:{port}")  # 打印监听信息

            conn, addr = sock.accept()  # 接受连接
            with conn:  # 确保连接资源释放
                print(f"Connected by {addr}")  # 打印连接方

                with open(output_file, 'wb') as f:  # 打开输出文件接收数据
                    while True:  # 循环接收数据直到结束
                        data = conn.recv(1024)  # 每次读取 1024 字节
                        if not data:  # 对端关闭后 recv 返回空
                            break  # 结束循环
                        f.write(data)  # 将数据写入文件

                print(f"Data received and saved to {output_file}")  # 输出保存成功信息
    except Exception as e:
        print(f"Error receiving data on {ip}:{port}: {e}")  # 打印异常信息

