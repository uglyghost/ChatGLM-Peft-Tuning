from transformers import AutoTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration
import torch
from peft import get_peft_model, LoraConfig, TaskType
from arguments import get_args


# 定义主函数
def main():
    # 解析命令行参数并赋值给args变量
    args = get_args()

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, device_map='auto')

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    model = get_peft_model(model, peft_config)
    model.load_state_dict(torch.load(args.peft_path), strict=False)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

    # instructions = json.load(open("data/alpaca_data.json"))

    instructions = [
        {
            'instruction': "西南财经大学的校长是谁？",
            "output": "西南财经大学的校长是卓志。他于2018年1月开始担任这一职务，并且是经济学博士和教授。他主要从事商业保险、风险管理和精算等领域的研究和高教管理。",
        },
        {
            'instruction': "你是谁？",
            "output": "我叫FinMoss，是金融智能问答的语言模型助手。",
        },
        {
            'instruction': "你作为金融领域的大语言模型可以做什么事情？",
            "output": "我作为金融领域的大语言模型，可以做很多事情，比如：\n1.回答您有关金融知识、产品、服务、政策等方面的问题。\n2.为您提供最新的金融资讯和数据分析，帮助您了解市场动态和投资机会。\n3.生成金融相关的文本，如新闻摘要、股评、基金报告等。\n4.与您进行有趣的对话，分享一些金融趣闻和小知识。",
        }
    ]

    answers = []

    with torch.no_grad():
        for idx, item in enumerate(instructions[:5]):
            input_text = f"### {idx+1}.Instruction:\n{item['instruction']}\n\n"
            if item.get('input'):
                input_text += f"### {idx+1}.Input:\n{item['input']}\n\n"
            input_text += f"### {idx+1}.Response:"
            batch = tokenizer(input_text, return_tensors="pt")
            out = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=torch.ones_like(batch["input_ids"]).bool(),
                max_length=args.max_length,
                temperature=args.temperature
            )
            out_text = tokenizer.decode(out[0])
            answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
            item['infer_answer'] = answer
            print(out_text)
            print(f"### {idx+1}.Answer:\n", item.get('output'), '\n\n')
            answers.append({'index': idx, **item})


if __name__ == "__main__":
    main()