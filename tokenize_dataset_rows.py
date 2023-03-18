# 导入json和argparse模块
import json
from arguments import get_args

# 随机打乱样本数据工具
import random

# 导入tqdm模块用于显示进度条
import tqdm.auto as tqdm

# 导入transformers和datasets模块
import datasets
import transformers


# 定义一个函数read_jsonl，接收一个路径参数path
def read_jsonl(path):
    # 以读取模式打开文件
    with open(path, "r", encoding='utf-8') as f:
        # 遍历文件中的每一行
        for line in f:
            # 将每一行转换为json对象并返回
            yield json.loads(line)


# 定义一个函数main，不接收任何参数
def main():
    # 解析命令行参数并赋值给args变量
    args = get_args()

    # 从预训练模型"THUDM/chatglm-6b"加载分词器tokenizer，并信任远程代码
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b", trust_remote_code=True
    )

    # 创建一个空列表all_tokenized用于存储分词后的结果
    all_tokenized = []

    # 遍历read_jsonl函数返回的json对象，并使用tqdm显示进度条
    for elem in tqdm.tqdm(read_jsonl(args.jsonl_path)):
        # 对每个json对象中的"text"字段进行分词，并截断到最大长度args.max_seq_length，并将结果添加到all_tokenized列表中
        all_tokenized.append(
            tokenizer.encode(
                elem["text"], max_length=args.max_seq_length, truncation=True,
            )
        )

    # 对all_tokenized列表进行随机打乱
    random.shuffle(all_tokenized)

    # 从all_tokenized列表创建一个数据集对象ds，并使用"input_ids"作为键名
    ds = datasets.Dataset.from_dict({"input_ids": all_tokenized})

    # 将数据集对象ds保存到args.save_path指定的路径下
    ds.save_to_disk(args.save_path)

    # 打印生成了多少个样本
    print(f"Generated {len(all_tokenized)} samples.")


if __name__ == "__main__":
    main()