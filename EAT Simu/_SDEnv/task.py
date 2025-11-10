import random  # 导入随机库用于生成任务属性
import numpy as np  # 导入 numpy 以便构造特征向量
import json  # 导入 json 以便进行序列化和反序列化


class Task:
    """
    表示单个 AIGC 任务的状态容器。

    参数说明：
    state_dim: 状态维度占位（当前未直接使用，保留兼容性）。
    duration: 任务预计执行时长（单位：秒）。
    arrival_time: 任务到达调度系统的时间戳。
    task_id: 任务唯一编号。
    node_ids: 计划分配的节点编号列表。
    co_num: 协同执行的节点数量。
    reload: 是否需要重新加载模型权重。
    execute: 是否允许执行该任务。
    steps: Stable Diffusion 推理步数。
    size: 输出图像总像素数，用于估算工作量。
    valid: 是否有效（例如资源不足时会标记为 False）。
    info: 附加信息字典，常用于存放 prompt、负向 prompt 等字段。
    """

    vector_len = 3  # 任务特征向量长度，供环境拼接观察值使用

    def __init__(self,
                 state_dim=1,
                 duration: float = 0,
                 arrival_time: float = 0,
                 task_id: int = -1,
                 node_ids: list[int] = [],  # 注意：默认空列表会在实例间共享，使用时应谨慎
                 co_num=1,
                 reload=False,
                 execute=True,
                 steps=30,
                 size=512 * 512,
                 valid=True,
                 info: dict = {}):
        self.duration = duration  # 记录任务预计执行时长
        self.arrival_time = arrival_time  # 记录任务到达时间
        self.task_id = task_id  # 保存任务编号
        self.node_ids = node_ids  # 保存协同节点列表
        self.co_num = co_num  # 协同节点数量
        self.reload = reload  # 是否需要重新加载模型
        self.steps = steps  # 推理步数
        self.size = size  # 图像像素数量
        self.info = info  # 附加信息字典
        self.execute = execute  # 是否被执行
        self.valid = valid  # 是否有效
        self.start_time = -1  # 任务实际开始时间（初始化为未开始）
        self.load_time = 0  # 记录加载模型耗时

    @property
    def vector(self):
        """
        返回任务特征向量（到达时间、归一化像素大小、协同节点数）。
        """
        return np.hstack([self.arrival_time, self.size / (512 * 512), self.co_num])

    def to_json(self):
        """
        将任务对象转换为 JSON 字符串，便于网络传输或存档。
        """
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str):
        """
        根据 JSON 字符串重建 Task 实例。

        参数说明：
        json_str: 序列化后的任务描述字符串。
        """
        data = json.loads(json_str)  # 将 JSON 转换为字典
        return cls(**data)  # 拆包字典并实例化


class TaskGenerator:
    """
    任务生成器，用于模拟新的 AIGC 请求到达。

    参数说明：
    fixed_steps: 是否固定推理步数（True 则随机生成步数，False 时由策略决定）。
    fixed_co_num: 是否固定协同节点数（True 表示随机选择，False 表示由策略输出）。
    fixed_size: 是否固定图像尺寸（True 固定 512x512）。
    max_steps: 固定步数模式下的最大推理步数。
    co_num: 可选的协同节点数量列表。
    size_option: 当不固定尺寸时可选的边长列表。
    """

    def __init__(self,
                 fixed_steps,
                 fixed_co_num,
                 fixed_size=True,
                 max_steps=64,
                 co_num=[1, 2, 4, 8],
                 size_option=[512, 768, 1024]):
        self.fixed_steps = fixed_steps  # 标记是否自动生成步数
        self.fixed_co_num = fixed_co_num  # 标记是否自动选择协同节点数
        self.fixed_size = fixed_size  # 标记是否固定图像尺寸
        self.size_option = size_option  # 自定义的尺寸候选集合
        self.max_steps = max_steps  # 随机步数的上限
        self.co_num = co_num  # 可选的协同节点数集合
        self.task_cnt = 0  # 已生成的任务数量

    def reset(self):
        """重置任务计数器，一般在场景重启时调用。"""
        self.task_cnt = 0

    def get_new_task(self):
        """生成一个新的 Task 实例，并随机填充任务属性。"""
        task = Task()  # 创建空任务
        self.task_cnt += 1  # 更新计数
        task.task_id = self.task_cnt  # 为任务分配唯一编号
        if self.fixed_steps:
            task.steps = random.random() * self.max_steps  # 随机选择推理步数
        if self.fixed_co_num:
            task.co_num = random.choice(self.co_num)  # 随机选择协同节点数量
        if self.fixed_size:
            task.size = 512 * 512  # 固定图像尺寸
        else:
            task.size = random.choice(self.size_option) * random.choice(self.size_option)  # 随机选择宽高乘积

        task.info['prompt'] = "orange"  # 默认正向提示词
        task.info['ng_prompt'] = "man!"  # 默认负向提示词

        return task  # 返回创建好的任务


class TaskQueue:
    """
    任务队列，负责维护待调度任务列表并提供特征向量。

    参数说明：
    task_generator: TaskGenerator 实例，用于按需补充任务。
    visible_len: 环境可见的队列长度（超出部分不进入状态）。
    init_job_num: 初始化填充的任务数量。
    state_dim: 状态表示维度（1 表示平铺向量，2 表示二维矩阵）。
    """

    def __init__(self,
                 task_generator: TaskGenerator,
                 visible_len: int = 10,
                 init_job_num: int = 10,
                 state_dim=1):
        self.visible_len = visible_len  # 状态中展示的任务数量
        self.task_queue: list[Task] = []  # 实际任务列表
        for _ in range(init_job_num):  # 预先填充任务
            self.task_queue.append(task_generator.get_new_task())
        self.state_dim = state_dim  # 状态维度模式

    def empty(self):
        """判断任务队列是否为空。"""
        return len(self.task_queue) == 0

    def is_vaild_id(self, id):
        """判断给定索引是否在队列范围内。"""
        return id <= len(self.task_queue)

    def get_avg_waiting_time(self, current_time: float) -> float:
        """
        计算队首可见任务的平均等待时间。

        参数说明：
        current_time: 环境当前时间戳。
        """
        task_num = min(self.visible_len, len(self.task_queue))  # 可观察的任务数量
        if task_num == 0:
            return 0  # 队列为空直接返回 0
        waiting_time = sum((current_time - task.arrival_time) for task in self.task_queue[:task_num]) / task_num
        return waiting_time

    def remove_task_id(self, id=-1):
        """
        按索引删除队列中的任务。

        参数说明：
        id: 目标任务在队列中的索引。
        """
        if not self.is_vaild_id(id):
            assert ("task index error! not in queue")
        self.task_queue.pop(id)  # 删除指定位置的任务

    def remove_task(self, task: Task):
        """
        按对象删除任务。

        参数说明：
        task: 需要移除的任务对象。
        """
        if task not in self.task_queue:
            assert ("task not in queue")
        self.task_queue.remove(task)  # 移除任务

    @property
    def vector(self):
        """
        提供队列的特征表示，供强化学习环境观察。
        """
        if self.state_dim == 1:
            if self.visible_len <= len(self.task_queue):
                return np.hstack(np.ravel([task.vector for task in self.task_queue[:self.visible_len]]))
            return np.hstack([
                np.ravel([task.vector for task in self.task_queue]),
                [0] * Task.vector_len * (self.visible_len - len(self.task_queue))
            ])
        if self.state_dim == 2:
            if self.visible_len <= len(self.task_queue):
                return np.stack([task.vector for task in self.task_queue[:self.visible_len]])
            padding = np.zeros((self.visible_len - len(self.task_queue), Task.vector_len))
            if len(self.task_queue) == 0:
                return padding  # 队列为空则直接返回零矩阵
            task_vector = np.array([task.vector for task in self.task_queue])
            return np.vstack([task_vector, padding])
                
