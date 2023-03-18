from transformers import Trainer
from modeling_chatglm import ChatGLMForConditionalGeneration
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import os
from arguments import get_args


@dataclass  # 使用dataclass装饰器，自动生成__init__等特殊方法
class FinetuneArguments:  # 定义一个数据类，用于存储微调的参数
    dataset_path: str = field(default="data/alpaca")  # 数据集路径，默认为"data/alpaca"
    model_path: str = field(default="output")  # 模型路径，默认为"output"
    lora_rank: int = field(default=8)  # LoRA的秩，默认为8


class CastOutputToFloat(nn.Sequential):   # 定义一个继承自nn.Sequential的类，用于将输出转换为浮点类型
    def forward(self, x): return super().forward(x).to(torch.float32)   # 重写forward方法，调用父类的forward方法，并将结果转换为torch.float32类型


class ModifiedTrainer(Trainer):  # 定义一个继承自Trainer的类，用于修改计算损失函数

    def compute_loss(self, model, inputs, return_outputs=False):  # 重写compute_loss方法，输入模型和输入数据
        return model(  # 返回模型的输出
            input_ids=inputs["input_ids"],  # 输入id
            attention_mask=torch.ones_like(inputs["input_ids"]).bool(),  # 注意力掩码，全1矩阵
            labels=inputs["input_ids"],  # 标签和输入id相同
        ).loss  # 输出损失值


def data_collator(features: list) -> dict:  # 定义一个函数，用于将特征列表转换为字典格式
    return {
        "input_ids": torch.stack([  # 返回一个键为"input_ids"的字典，值为特征列表中每个元素的"input_ids"属性组成的张量堆叠
            torch.LongTensor(f["input_ids"])
            for f in features
        ])
    }


def save_tunable_parameters(model, path):  # 定义一个函数，用于保存模型中可调节的参数到指定路径
    saved_params = {  # 创建一个字典，存储模型中需要梯度的参数
        k: v.to("cpu")  # 将参数值转换为cpu类型
        for k, v in model.named_parameters()  # 遍历模型中命名的参数
        if v.requires_grad  # 如果参数需要梯度
    }
    torch.save(saved_params, path)  # 使用torch.save函数，将字典保存到路径


# 定义主函数
def main():
    # 解析命令行参数并赋值给args变量
    args = get_args()

    # 从预训练模型"THUDM/chatglm-6b"加载模型，并设置一些参数
    model = ChatGLMForConditionalGeneration.from_pretrained(
        "THUDM/chatglm-6b",
        load_in_8bit=True,  # 使用8位精度加载模型，节省内存
        trust_remote_code=True,  # 信任远程代码，允许执行自定义操作
        device_map='auto')  # 自动分配设备映射

    if args.checkpoint_enable:
        model.gradient_checkpointing_enable()  # 启用梯度检查点功能，减少内存占用

    if args.grads_enable:
        model.enable_input_require_grads()  # 启用输入梯度计算功能，支持高阶导数

    model.is_parallelizable = True  # 设置模型为可并行化
    model.model_parallel = True  # 设置模型为并行模式
    model.lm_head = CastOutputToFloat(model.lm_head)  # 将输出层转换为浮点类型，提高精度
    model.config.use_cache = False  # 关闭缓存功能

    # 设置peft配置，包括任务类型、推理模式、秩、alpha值和dropout率等参数
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # 获取peft模型，即使用低秩逼近技术优化后的模型
    model = get_peft_model(model, peft_config)
    if args.continue_training:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "chatglm-lora.pt")), strict=False)

    # 从指定路径加载数据集，并转换为torch格式
    dataset = datasets.load_from_disk(args.dataset_path)

    # 开始训练过程，使用ModifiedTrainer类创建训练器对象，并传入模型、数据集、参数和数据整理器等参数
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=args,
        data_collator=data_collator,

    )
    trainer.train()  # 调用train方法进行训练

    # 保存训练后的模型参数到指定路径下的文件中
    save_tunable_parameters(model, os.path.join(args.output_dir, "chatglm-lora.pt"))


if __name__ == "__main__":
    main()