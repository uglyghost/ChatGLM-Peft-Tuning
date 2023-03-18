# ChatGLM-Peft-Tuning

该项目基于清华的 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 进行finetune.
基于项目 [mymusise](https://github.com/mymusise/ChatGLM-Tuning) 修改


特别鸣谢！


## 测试环境

- 显卡: GTX 3090 (24G) & A100 (40G)
- 系统: Windows 11 & Ubuntu 18.04
- 建议: python >=3.8, CUDA 11.2+
- 环境: pip install -r requirements.txt

### Windows 注意 
- Windows环境运行train.py时，bitsandbytes会报错:

  `CUDA Setup failed despite GPU being available. Inspect the CUDA SETUP outputs above to fix your environment!`


- bitsandbytes: 轻量级的CUDA自定义函数的包装器，主要提供了8位优化器、矩阵乘法（LLM.int8()）和量化函数12。它可以用于PyTorch框架，提高深度学习模型的训练速度和效率2。


- 解决方法:
  1. put `libbitsandbytes_cuda116.dll` in 
      
     `C:\Users\xxx\miniconda3\envs\textgen\lib\site-packages\bitsandbytes\`
  2. edit `\bitsandbytes\cuda_setup\main.py`. search for:
  
     `if not torch.cuda.is_available(): return 'libsbitsandbytes_cpu.so', None, None, None, None`
     
     replace with:
     
     `
     if torch.cuda.is_available(): return 'libbitsandbytes_cuda116.dll', None, None, None, None 
     `
  3. search for this twice:
  
      `self.lib = ct.cdll.LoadLibrary(binary_path)`
  
      replace with:
      
      `self.lib = ct.cdll.LoadLibrary(str(binary_path))`


### 项目概述
- data
  - alpaca
    - ...
  - data.json
- output
  - checkpoint
    - ...  
  - chatglm-lora.pt
  - ...
- tokenize_dataset_rows.py
- finetune.py
- infer.py
- ...

## Pretreatment

```bash
python tokenize_dataset_rows.py
```

配置参数见 `arguments.py` '''tokenize_dataset_rows.py 参数配置'''
- `--jsonl_path` 微调的数据路径, 格式jsonl, 对每行的['text']字段进行encode
- `--save_path`  用于训练数据集的存储路径
- `--max_seq_length` 样本文本的最大长度

## Finetune

```bash
python finetune.py --save_total_limit 2  --dataset_path data/alpaca --lora_rank 8 --per_device_train_batch_size 1  --gradient_accumulation_steps 1 --max_steps 52000 --save_steps 1000 --learning_rate 2e-5 --logging_steps 50 --output_dir output
```

配置参数见 `arguments.py` '''finetune.py 参数配置''' 
- `--dataset_path` 字符串集合的json格式的数据集
- `--per_device_train_batch_size` 每个设备上的数据批次，显存足够可增加
- `--gradient_accumulation_steps` 多次计算得到的梯度值进行累加，一次性进行参数更新
- `--max_steps` 最大训练迭代次数
- `--save_steps` checkpoint保存步长
- `--save_total_limit` 保存条目数量上限
- `--learning_rate` 模型学习率
- `--logging_steps` 日志输出间隔
- `--output_dir` finetune模型 & checkpoint 存储目录
- `--fp16` 

LORA

- `--lora_rank` 低秩矩阵的秩
- `--lora_alpha` 控制低秩矩阵和原始矩阵之间权重平衡的系数
- `--lora_dropout` 防止过拟合


# Infer
```bash
python infer.py
```

配置参数见 `arguments.py` '''infer.py 参数配置'''
- `--peft_path` finetune模型存储地址
- `--max_length` 最大输出长度
- `--temperature` 情感

# Datasets
- 数据集: [alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- 中文财经类QA数据集

# TODO:

- 数据/模型/张量并行 使用GLM pretrain 和 finetune 实现？
- 使用RLHF 参考 [trlx](https://github.com/CarperAI/trlx) 


## Cite

清华开源项目，参考引用下列论文

```
@inproceedings{
  zeng2023glm-130b,
  title={{GLM}-130B: An Open Bilingual Pre-trained Model},
  author={Aohan Zeng and Xiao Liu and Zhengxiao Du and Zihan Wang and Hanyu Lai and Ming Ding and Zhuoyi Yang and Yifan Xu and Wendi Zheng and Xiao Xia and Weng Lam Tam and Zixuan Ma and Yufei Xue and Jidong Zhai and Wenguang Chen and Zhiyuan Liu and Peng Zhang and Yuxiao Dong and Jie Tang},
  booktitle={The Eleventh International Conference on Learning Representations (ICLR)},
  year={2023},
  url={https://openreview.net/forum?id=-Aw0rrrPUF}
}
```

```
@inproceedings{du2022glm,
  title={GLM: General Language Model Pretraining with Autoregressive Blank Infilling},
  author={Du, Zhengxiao and Qian, Yujie and Liu, Xiao and Ding, Ming and Qiu, Jiezhong and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={320--335},
  year={2022}
}
```