import numpy as np  # 导入 numpy 用于数值计算
import json  # 导入 json 用于序列化与反序列化

class Task():
    vector_len = 3  # 定义任务特征向量长度

    def __init__(self,
                 state_dim = 1,
                 duration: float = 0,
                 arrival_time: float = 0,
                 task_id: int = -1,
                 node_ids: list[int] = [],
                 co_num=1,
                 reload=False,
                 execute = True,
                 steps=30,
                 size=512*512,
                 valid = True,
                 real_arrival_time = 0,
                 finish_time = 0,
                 start_time = 0,
                 load_time = 0,
                 info: dict={},**kwargs):
        """
        参数说明（全部为中文解释）：
        state_dim: 状态维度参数，默认 1，用于兼容外部状态描述。
        duration: 任务预计执行时长。
        arrival_time: 任务在调度系统中的到达时间。
        task_id: 任务唯一标识符。
        node_ids: 可参与任务协作的节点编号列表。
        co_num: 协同节点数量。
        reload: 是否复用既有模型权重或上下文。
        execute: 是否实际执行该任务。
        steps: 文生图推理步数。
        size: 输出图像像素总数。
        valid: 标记任务是否有效。
        real_arrival_time: 实际到达时间。
        finish_time: 实际完成时间。
        start_time: 实际开始时间。
        load_time: 模型加载耗时。
        info: 附加信息字典，例如提示词等。
        kwargs: 预留的其他动态参数。
        """
        self.duration = duration  # 保存预计执行时长
        self.arrival_time = arrival_time  # 保存到达时间
        self.task_id = task_id  # 保存任务 ID
        self.node_ids = node_ids  # 保存协作节点列表
        self.co_num = co_num  # 保存协作节点数量
        self.reload = reload  # 标记是否需要重新加载
        self.steps = steps  # 保存推理步数
        self.size = size  # 保存生成图像像素大小
        self.info = info  # 保存附加信息
        self.execute = execute  # 标记是否执行任务
        self.valid = valid  # 标记任务是否有效
        self.start_time = -1  # 初始化记录开始时间
        self.load_time = 0  # 初始化加载时间
        self.real_arrival_time = 0,  # 初始化实际到达时间（注意逗号导致成为元组）
        self.finish_time = 0,  # 初始化完成时间（注意逗号导致成为元组）
        self.start_time = 0,  # 初始化开始时间（再次赋值）

    @property
    def vector(self):
        # 返回任务的特征向量（到达时间、归一化大小、协同数量）
        return np.hstack([self.arrival_time, self.size/(512*512), self.co_num])

    def to_json(self):
        # 将对象属性字典序列化为 JSON 字符串
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str):
        # 将 JSON 字符串反序列化为字典
        data = json.loads(json_str)
        # 拆包字典并创建新的 Task 实例
        return cls(**data)

class TaskDistriConfig:
    def __init__(self, node_ips: list[str], torch_port: str, master_ip: str, master_res_port: str, node_id:str):
        """
        参数说明：
        node_ips: 所有协作节点的 IP 列表。
        torch_port: torchrun 使用的通信端口。
        master_ip: 主节点 IP 地址。
        master_res_port: 主节点接收结果的端口。
        node_id: 当前节点的编号（字符串形式）。
        """
        self.node_ips = node_ips  # 保存节点 IP 列表
        self.torch_port = str(torch_port)  # 记录 torchrun 端口
        self.master_ip = master_ip  # 记录主节点 IP
        self.master_res_port = str(master_res_port)  # 记录主节点结果端口
        self.main_node_ip = self.node_ips[0] if self.node_ips else None  # 获取主节点 IP
        self.node_id = str(node_id)  # 记录当前节点 ID

    def to_json(self):
        # 将所有属性转换为 JSON 方便网络传输
        return json.dumps({
            "node_ips": self.node_ips,
            "torch_port": self.torch_port,
            "master_ip": self.master_ip,
            "master_res_port": self.master_res_port,
            "main_node_ip": self.main_node_ip,
            "node_id":self.node_id
        })

    def __eq__(self, other):
        # 比较两个配置是否相同
        if not isinstance(other, TaskDistriConfig):
            return False  # 类型不同直接返回 False
        return (
            self.node_ips == other.node_ips and
            self.torch_port == other.torch_port and
            self.master_ip == other.master_ip and
            self.master_res_port == other.master_res_port and
            self.node_id == other.node_id
        )

    @classmethod
    def from_json(cls, json_str):
        # 将 JSON 字符串转成字典
        data = json.loads(json_str)
        # 根据字典内容构建 TaskDistriConfig 实例
        return cls(
            node_ips=data["node_ips"],
            torch_port=data["torch_port"],
            master_ip=data["master_ip"],
            master_res_port=data["master_res_port"],
            node_id = data["node_id"]
        )

class StableDiffusionCommand:
    def __init__(self, task: Task, districonfig: TaskDistriConfig):
        """
        参数说明：
        task: 需要执行的任务对象。
        districonfig: 分布式执行配置对象。
        """
        self.task = task  # 保存任务描述
        self.districonfig = districonfig  # 保存分布式配置

    def to_json(self):
        # 分别序列化 Task 与 DistriConfig 再组合成 JSON
        return json.dumps({
            "task": json.loads(self.task.to_json()),         # 确保嵌套为字典
            "districonfig": json.loads(self.districonfig.to_json())
        })

    @classmethod
    def from_json(cls, json_str):
        # 将 JSON 字符串解析为字典
        data = json.loads(json_str)
        # 依次反序列化 Task 与 DistriConfig
        task = Task.from_json(json.dumps(data["task"]))
        districonfig = TaskDistriConfig.from_json(json.dumps(data["districonfig"]))
        # 创建 StableDiffusionCommand 实例并返回
        return cls(task=task, districonfig=districonfig)
