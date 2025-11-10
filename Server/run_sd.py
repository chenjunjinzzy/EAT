import torch  # 导入 PyTorch 以执行张量与分布式操作
from distrifuser.pipelines import DistriSDPipeline  # 导入分布式 Stable Diffusion 管线
from distrifuser.utils import DistriConfig  # 导入分布式配置工具
from template import StableDiffusionCommand  # 导入命令对象
from tools import receive_task, send_result  # 导入结果收发工具函数
import torch.distributed as dist  # 导入 PyTorch 分布式通信模块
import sys  # 访问命令行参数
import gc  # 控制垃圾回收释放显存

def get_cmd()->StableDiffusionCommand:
    """
    功能：读取命令行参数中的 JSON 字符串并构造 StableDiffusionCommand 对象。
    无参数：直接使用 sys.argv[1]。
    返回：StableDiffusionCommand 实例。
    """
    return StableDiffusionCommand.from_json(sys.argv[1])  # 将 JSON 文本还原为命令对象

def run_sd():
    """
    功能：执行分布式 Stable Diffusion 推理循环，接收任务、生成图片并上报结果。
    无显式参数：依赖命令行传入的 StableDiffusionCommand 作为初始任务。
    """
    cmd = get_cmd()  # 读取首个任务命令
    distri_config = DistriConfig(height=1440, width=1440, warmup_steps=4, mode="stale_gn")  # 构建分布式配置
    pipeline = DistriSDPipeline.from_pretrained(  # 按配置加载预训练模型
        distri_config = distri_config,  # 传入分布式配置
        pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4",  # 使用官方权重
        local_files_only = True,  # 强制使用本地缓存
    )

    while True:  # 持续处理任务直至配置变化
        image = pipeline(
            prompt=cmd.task.info['prompt'],  # 使用正向提示词
            negative_prompt=cmd.task.info['ng_prompt'],  # 使用反向提示词
            generator=torch.Generator(device="cuda").manual_seed(233),  # 固定种子确保可重复性
            num_inference_steps = cmd.task.steps  # 控制推理步数
        ).images[0]  # 获取生成的首张图片
        file_name = f"{cmd.task.info['prompt'].replace(' ', '_')}__{cmd.task.info['ng_prompt'].replace(' ', '_')}__{cmd.task.steps}.png"  # 按提示词命名文件
        image.save(file_name)  # 保存图片到本地
        if cmd.districonfig.node_id == "0":  # 仅主节点负责回传结果
            send_result(cmd.districonfig.master_ip, cmd.districonfig.master_res_port, file_name=file_name)  # 回传生成结果
        if len(cmd.districonfig.node_ips) > 1:  # 多节点时需要同步
            dist.barrier()  # 等待所有节点完成当前任务
        print("task finish!")  # 打印任务完成日志
        new_cmd = receive_task(16122)  # 等待下一任务
        print(f"old districonfig:{cmd.task.node_ids}")  # 打印旧协作节点集合
        print(f"new dustriconfig:{new_cmd.task.node_ids}")  # 打印新协作节点集合
        if cmd.task.node_ids != new_cmd.task.node_ids:  # 若节点集合发生变化
            if len(cmd.districonfig.node_ips) > 1:  # 多节点需要同步退出
                print("waiting for other node")  # 日志提示
                dist.barrier()  # 等待其他节点
            print("start destroy group")  # 提示进入销毁阶段
            torch.cuda.synchronize()  # 等待 CUDA 操作完成
            del pipeline  # 删除管线释放显存
            gc.collect()  # 手动触发垃圾回收
            dist.destroy_process_group()  # 销毁分布式进程组
            print("sub process finish!")  # 打印结束信息
            return  # 退出循环结束子进程
        cmd = new_cmd  # 未变化时继续使用新任务

run_sd()  # 直接运行推理主循环
