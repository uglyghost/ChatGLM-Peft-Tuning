from transformers import AutoTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration
import torch
from peft import get_peft_model, LoraConfig, TaskType, tuners
from arguments import get_args
from single_layer import single_layer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 定义主函数
def main():
    # 解析命令行参数并赋值给args变量
    args = get_args()

    # reload the model
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True,
                                                            device_map='auto')

    # 设置peft配置，包括任务类型、推理模式、秩、alpha值和dropout率等参数
    peft_config = LoraConfig(
        peft_type="LORA",
        task_type="SEQ_2_SEQ_LM",
        # inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q", "k", "v"]
    )
    # convert it again
    for key, module in model.named_modules():
        if key.endswith('attention'):
            try:
                qkv_layer = single_layer(module.query_key_value.in_features, module.query_key_value.out_features)
                qkv_layer.update(module.query_key_value)
                module.query_key_value = qkv_layer
            except:
                print('no')
                pass
            module.query_key_value = tuners.lora.LoraModel(peft_config, module.query_key_value)

    # load the LoRA checkpoint
    model.load_state_dict(torch.load('output_finetune_99.pt'), strict=False)

    model.half().cuda().eval()

    # Let's chat!
    '''
    response, history = model.chat(tokenizer, "你是谁？", history=[])
    print(response)
    response, history = model.chat(tokenizer, "西南财经大学副校长是谁？", history=[])
    print(response)
    response, history = model.chat(tokenizer, "西南财经大学校长是谁？", history=[])
    print(response)
    '''

    # return

    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, device_map='auto')
    '''
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
    '''
    # instructions = json.load(open("data/alpaca_data.json"))

    instructions = [
        {
            'instruction': "西南财经大学校长是谁？",
            "output": "西南财经大学的校长是卓志。他于2018年1月开始担任这一职务，并且是经济学博士和教授。他主要从事商业保险、风险管理和精算等领域的研究和高教管理。",
        },
        {
            'instruction': "西南财经大学副校长是谁？",
            "output": "西南财经大学现有两位副校长，分别是张邦富1和李志生。张邦富是党委常委、副校长，主要负责学校的教学、科研、人才培养等工作。李志生是党委常委、副校长，于2022年7月28日正式任职，主要负责学校的发展规划、国际合作与交流等工作。",
        }
    ]

    answers = []

    with torch.no_grad():
        for idx, item in enumerate(instructions[:5]):
            input_text = f"### {idx+1}.指令:\n{item['instruction']}\n\n"
            if item.get('input'):
                input_text += f"### {idx+1}.输入:\n{item['input']}\n\n"
            input_text += f"### {idx+1}.回复:"
            # print(input_text)
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
            # print(f"### {idx+1}.Answer:\n", item.get('output'), '\n\n')
            answers.append({'index': idx, **item})


if __name__ == "__main__":
    main()