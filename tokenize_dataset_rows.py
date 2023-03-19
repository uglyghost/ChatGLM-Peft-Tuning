import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from arguments import get_args


class AlpacaDataset(Dataset):
    def __init__(self, pairs, tokenizer, device) -> None:
        super().__init__()
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.device = device
        self.EOS_ID = 150005

    def __getitem__(self, index):
        prompt = self.tokenizer.encode(self.pairs[index]['prompt'])
        completion = self.tokenizer.encode(self.pairs[index]['completion'], add_special_tokens=False) + [self.EOS_ID]

        seq = prompt + completion
        context_length = seq.index(150004) + 1

        attention_mask = torch.ones((len(seq), len(seq)), device=self.device )
        attention_mask.tril_()
        attention_mask[..., :context_length - 1] = 1
        attention_mask.unsqueeze_(0)
        attention_mask = (attention_mask < 0.5).bool()

        position_ids = torch.stack([torch.arange(0, len(seq), device=self.device ), torch.concat(
            [torch.zeros(context_length - 2, device=self.device ),
             torch.arange(0, len(seq) - context_length + 2, device=self.device )])]).long()
        labels = torch.tensor([-100] * len(prompt) + completion, device=self.device ).long()

        return {'input_ids': seq, 'attention_mask': attention_mask, "labels": labels, 'position_ids': position_ids}

    def __len__(self):
        return len(self.pairs)


def collate_fn(batch):
    input_ids = []
    attention_mask = []
    labels = []
    position_ids = []
    # TODO: padding for batch training
    for obj in batch:
        input_ids.append(obj['input_ids'])
        attention_mask.append(obj['attention_mask'])
        labels.append(obj['labels'])
        position_ids.append(obj['position_ids'])
    return {'input_ids': torch.tensor(input_ids).long(),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels),
            'position_ids':torch.stack(position_ids)}


# 定义一个函数main，不接收任何参数
def load_dataset():
    # 解析命令行参数并赋值给args变量
    args = get_args()

    device = 'cuda'

    PROMPT_DICT = {
        "prompt_input": (
            #"下面的指令介绍了一个任务问题，并且提供了上下文的输入。"
            #"请写一个合适的回复，回复指令中描述的问题。\n\n"
            "### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 回复:"
        ),
        "prompt_no_input": (
            #"下面的指令介绍了一个任务问题。"
            #"请写一个合适的回复，回复指令中描述的问题。\n\n"
            "### 指令:\n{instruction}\n\n### 回复:"
        )
    }

    with open(args.jsonl_path, 'r') as f:
        content = json.load(f)

    pairs = []

    for line in content:
        if line['input'] == '':
            prompt = PROMPT_DICT['prompt_no_input'].format_map(line)
        else:
            prompt = PROMPT_DICT['prompt_input'].format_map(line)
        completion = line['output']
        pairs.append({'prompt': prompt, 'completion': completion})

    # 从预训练模型"THUDM/chatglm-6b"加载分词器tokenizer，并信任远程代码
    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b", trust_remote_code=True
    )

    train_dataset = AlpacaDataset(pairs, tokenizer=tokenizer, device=device)
    # print(pairs)
    train_dataloader = DataLoader(dataset=train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=1)

    # 打印生成了多少个样本
    print(f"Generated {len(train_dataloader.dataset)} samples.")

    return train_dataloader