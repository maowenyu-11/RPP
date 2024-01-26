>📋  A  README.md for code accompanying our paper RPP

# Reinforced Prompt Personalization for Recommendation with Large Language Models


## 1. Prepare the environment

 Prepare the environment to evaluate on ChatGPT and LLaMa2-7b-chat
```
cd RPP
pip install -r requirements.txt
```
 Prepare the environment to evaluate on Alpaca
```
pip install -r requirements_A.txt
```
## 2. Prepare the pre-trained LLMs

Insert the API of gpt-3.5-turbo to openai_api.yaml to test the performance of ChatGPT

Download the pre-trained Bert model to transform prompts into embedding(<https://huggingface.co/bert-base-chinese>) 

Download the pre-trained LLaMa2-7B-chat model (<https://ai.meta.com/llama/>)

Download the pre-trained huggingface model of LLaMA2-7B to finetune Alpaca (<https://huggingface.co/meta-llama/Llama-2-7b-hf>)

## 2. Prepare the checkpoints of RPP and finetuned Alpaca
Put the checkpoints of RPP to the dir path saved/ and the checkpoints of finetuned Alpaca to the dir path model/

## Training

We use the NVIDIA A100 to conduct all our experiments on LLMs:

For the dataset of ML-1M and Lastfm, the initial interaction history length (few-shot) ```ini_len: 1```

For the dataset of Games, the initial interaction history length (few-shot) ```ini_len: 5```

To train the frozen LLM of LLaMa2-7B-chat:
```
selected_user_suffix: train
CUDA_VISIBLE_DEVICES=[] torchrun --nproc_per_node 1 --master_port=[] train.py

```
To evaluate the frozen LLM of LLaMa2-7B-chat:
```
selected_user_suffix: test
CUDA_VISIBLE_DEVICES=[] torchrun --nproc_per_node 1 --master_port=[] evaluate.py
```



