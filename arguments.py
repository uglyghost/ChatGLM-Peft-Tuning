# 导入argparse模块
import argparse

# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description="ChatGLM model fine-tuning")

'''tokenize_dataset_rows.py 参数配置'''
# 对微调数据集做预处理，tokenized -> convert to json -> save to binary file.
parser.add_argument("--jsonl_path", type=str, default="data/change_name.jsonl.txt")  # json格式的数据集
parser.add_argument("--save_path", type=str, default="data/alpaca")                  # 用于训练数据集的存储路径
parser.add_argument("--max_seq_length", type=int, default=512)                       # 样本文本的最大长度

'''finetune.py 参数配置'''
# training
parser.add_argument("--continue_training", type=bool, default=True)                  # 是否在微调模型上继续训练
parser.add_argument("--checkpoint_enable", type=bool, default=True)                  # 是否开启checkpoint功能
parser.add_argument("--grads_enable", type=bool, default=True)                       # 启用输入梯度计算功能，支持高阶导数

# LoRA是一种低秩适应大型语言模型的方法
parser.add_argument("--lora_rank", type=int, default=8)                              # 低秩矩阵的秩
parser.add_argument("--lora_alpha", type=int, default=32)                            # 控制低秩矩阵和原始矩阵之间权重平衡的系数
parser.add_argument("--lora_dropout", type=float, default=0.1)                       # 防止过拟合的概率

parser.add_argument("--dataset_path", type=str, default="data/alpaca")               # 字符串集合的json格式的数据集
parser.add_argument("--per_device_train_batch_size", type=int, default=1)            # 每个设备上的数据批次，显存足够可增加
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)            # 多次计算得到的梯度值进行累加，一次性进行参数更新
parser.add_argument("--max_steps", type=int, default=10000)                          # 最大训练迭代次数
parser.add_argument("--save_steps", type=int, default=1000)                          # checkpoint保存步长
parser.add_argument("--save_total_limit", type=int, default=2)                       # 保存条目数量上限
parser.add_argument("--learning_rate", type=float, default=2e-5)                     # 模型学习率
parser.add_argument("--logging_steps", type=int, default=50)                         # 日志输出间隔
parser.add_argument("--output_dir ", type=str, default='output')                     # finetune模型 & checkpoint 存储目录

'''infer.py 参数配置'''
parser.add_argument("--peft_path ", type=str, default='output/chatglm-lora.pt')       # finetune模型存储地址
parser.add_argument("--max_length", type=int, default=512)                            # 最大输出长度
parser.add_argument("--temperature", type=int, default=0)                             # 情感

# 解析ArgumentParser对象，获得argparse.Namespace对象
args = parser.parse_args()


def get_args():
    return parser.parse_args()