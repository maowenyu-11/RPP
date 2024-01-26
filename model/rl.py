import os.path as osp
import torch
import openai
import time
import asyncio
import sys
import numpy as np
from tqdm import tqdm
from recbole.model.abstract_recommender import SequentialRecommender
import torch
import transformers
from peft import PeftModel
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils import dispatch_openai_requests, dispatch_single_openai_requests
# os.environ["CUDA_VISIBLE_DEVICES"] = '3' 
from typing import List, Optional
# from llama import Llama, Dialog
class Rl(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.max_tokens = config['max_tokens']
        self.api_model_name = config['api_name']
        openai.api_key = config['api_key']
        openai.api_base = config['api_base']
        self.api_batch = config['api_batch']
        self.async_dispatch = config['async_dispatch']
        self.temperature = config['temperature']

        self.max_his_len = config['max_his_len']
        self.recall_budget = config['recall_budget']
        self.boots = config['boots']
        self.data_path = config['data_path']
        self.dataset_name = dataset.dataset_name
        self.id_token = dataset.field2id_token['item_id']
        self.item_text = self.load_text()
        self.logger.info(
            f'Avg. t = {np.mean([len(_) for _ in self.item_text])}')

        self.fake_fn = torch.nn.Linear(1, 1)

    def load_text(self):
        token_text = {}
        item_text = ['[PAD]']
        feat_path = osp.join(self.data_path, f'{self.dataset_name}.item')
        if self.dataset_name == 'ml-1m':
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, movie_title, release_year, genre = line.strip(
                    ).split('\t')
                    token_text[item_id] = movie_title
            for i, token in enumerate(self.id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                if raw_text.endswith(', The'):
                    raw_text = 'The ' + raw_text[:-5]
                elif raw_text.endswith(', A'):
                    raw_text = 'A ' + raw_text[:-3]
                item_text.append(raw_text)
            return item_text
        elif self.dataset_name == 'lastfm':
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, name, url,picture_url = line.strip(
                    ).split('\t')
                    token_text[item_id] = name
            for i, token in enumerate(self.id_token):
                # print(self.id_token)
                if token == '[PAD]': continue
                raw_text = token_text[token]
                item_text.append(raw_text)
            return item_text
        
        elif self.dataset_name == 'Games':
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, title = line.strip().split('\t')
                    token_text[item_id] = title
            for i, token in enumerate(self.id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                item_text.append(raw_text)
            return item_text
        else:
            raise NotImplementedError()
    def evaluate(self,
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=1024,
        batch_size=10,
        **kwargs,
    ):
        
        prompt = self.generate_prompt(instruction, input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.config['device'])
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output =self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        # responses=[]
        # for i in range(batch_size):
        response=self.get_response(output)
        return response
    def generate_prompt(self,instruction, input=None):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:
    """
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

    ### Instruction:
    {instruction}

    ### Response:
    """
    def get_response(self, output):
        return output.split("### Response:")[1].strip()

    def predict_on_subsets(self, interaction, idxs):
        """
        Main function to rank with LLMs

        :param interaction:
        :param idxs: item id retrieved by candidate generation models [batch_size, candidate_size]
        :return:
        """
        base_model= "/data/maowy/llama-2-hf-7b"# decapoda-research/llama-13b-hf
        lora_weights= "/data/maowy/rlprec/finetune/lora-alpaca"# chansung/gpt4-alpaca-lora-13b、tloen/alpaca-lora-7b
        assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
        load_8bit: bool = False
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        self.model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    

    # unwind broken decapoda-research config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        if not load_8bit:
            self.model.half()  # seems to fix bugs for some users.

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
            
        origin_batch_size = idxs.shape[0]
        if self.boots:
            """ 
            bootstrapping is adopted to alleviate position bias
            `fix_enc` is invalid in this case"""
            idxs = np.tile(idxs, [self.boots, 1])
            np.random.shuffle(idxs.T)
        batch_size = idxs.shape[0]
        pos_items = interaction[self.POS_ITEM_ID]

        prompt_list = []
        openai_responses=[]
        for i in tqdm(range(batch_size)):
            user_his_text, candidate_text, candidate_text_order, candidate_idx = self.get_batch_inputs(
                interaction, idxs, i)

            prompt = self.construct_prompt(self.dataset_name, user_his_text,
                                           candidate_text_order)
            prompt_list.append( prompt)
   
        for i in range(batch_size):
          openai_responses.append(self.evaluate(prompt_list[i]))

        scores = torch.full((idxs.shape[0], self.n_items), -10000.)
        for i, openai_response in enumerate(tqdm(openai_responses)):
            user_his_text, candidate_text, candidate_text_order, candidate_idx = self.get_batch_inputs(
                interaction, idxs, i)

            response = openai_response
            response_list = response.split('\n')

            self.logger.info(prompt_list[i])
            self.logger.info(response)
            self.logger.info(f'Here are candidates: {candidate_text}')
            self.logger.info(f'Here are answer: {response_list}')

            if self.dataset_name == 'ml-1m':
                rec_item_idx_list = self.parsing_output_text(
                    scores, i, response_list, idxs, candidate_text)
            else:
                rec_item_idx_list = self.parsing_output_text(
                    scores, i, response_list, idxs, candidate_text)

            if int(pos_items[i % origin_batch_size]) in candidate_idx:
                target_text = candidate_text[candidate_idx.index(
                    int(pos_items[i % origin_batch_size]))]
                try:
                    ground_truth_pr = rec_item_idx_list.index(target_text)
                    self.logger.info(
                        f'Ground-truth [{target_text}]: Ranks {ground_truth_pr}'
                    )
                except:
                    self.logger.info(f'Fail to find ground-truth items.')
                    print(target_text)
                    print(rec_item_idx_list)
        if self.boots:
            scores = scores.view(self.boots, -1, scores.size(-1))
            scores = scores.sum(0)
        return scores

    def get_batch_inputs(self, interaction, idxs, i):
        user_his = interaction[self.ITEM_SEQ]
        user_his_len = interaction[self.ITEM_SEQ_LEN]
        origin_batch_size = user_his.size(0)
        real_his_len = min(self.max_his_len,
                           user_his_len[i % origin_batch_size].item())
        user_his_text = [str(j) + '. ' + self.item_text[user_his[i % origin_batch_size, user_his_len[i % origin_batch_size].item() - real_his_len + j].item()] \
                for j in range(real_his_len)]
        candidate_text = [
            self.item_text[idxs[i, j]] for j in range(idxs.shape[1])
        ]
        candidate_text_order = [
            str(j) + '. ' + self.item_text[idxs[i, j].item()]
            for j in range(idxs.shape[1])
        ]
        candidate_idx = idxs[i].tolist()

        return user_his_text, candidate_text, candidate_text_order, candidate_idx

    def construct_prompt(self, dataset_name, user_his_text,
                         candidate_text_order):
        if dataset_name == 'ml-1m':
            prompt = f"I've watched the following movies in the past in order:\n{user_his_text}\n\n" \
                    f"Now there are {self.recall_budget} candidate movies that I can watch next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break."
        elif dataset_name == 'lastfm':
            prompt = f"I've listened the following musics in the past in order:\n{user_his_text}\n\n" \
                    f"Now there are {self.recall_budget} candidate musics that I can consider to listen next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} musics by measuring the possibilities that I would like to listen next most, according to the given listening records. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break."
      
        elif dataset_name == 'Games':
            prompt = f"I've purchased the following products in the past in order:\n{user_his_text}\n\n" \
                    f"Now there are {self.recall_budget} candidate products that I can consider to purchase next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Please think step by step.\n" \
                    f"Please only output the order numbers after ranking. Split these order numbers with line break."
        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        return prompt

    def dispatch_openai_api_requests(self, prompt_list, batch_size):
        openai_responses = []
        self.logger.info('Launch OpenAI APIs')
        if self.async_dispatch:
            self.logger.info('Asynchronous dispatching OpenAI API requests.')
            for i in tqdm(range(0, batch_size, self.api_batch)):
                while True:
                    try:
                        openai_responses += asyncio.run(
                            dispatch_openai_requests(
                                prompt_list[i:i + self.api_batch],
                                self.api_model_name, self.temperature))
                        break
                    except Exception as e:
                        print(
                            f'Error {e}, retry batch {i // self.api_batch} at {time.ctime()}',
                            flush=True)
                        time.sleep(20)
        else:
            self.logger.info('Dispatching OpenAI API requests one by one.')
            for message in tqdm(prompt_list):
                while True:
                    try:
                        openai_responses.append(
                            dispatch_single_openai_requests(
                                message, self.api_model_name,
                                self.temperature))
                        break
                    except Exception as e:
                        print(f'Error {e}, retry ', flush=True)
                        time.sleep(20)
                # time.sleep(10)
        self.logger.info('Received OpenAI Responses')
        return openai_responses

    def parsing_output_text(self, scores, i, response_list, idxs,
                            candidate_text):
        rec_item_idx_list = []
        found_item_cnt = 0
        for j, item_detail in enumerate(response_list):
            if len(item_detail) < 1:
                continue
            if item_detail.endswith('candidate movies:'):
                continue
            pr = item_detail.find('. ')
            if item_detail[:pr].isdigit():
                item_name = item_detail[pr + 2:]
            else:
                item_name = item_detail

            matched_name = None
            for candidate_text_single in candidate_text:
                if candidate_text_single in item_name:
                    if candidate_text_single in rec_item_idx_list:
                        break
                    rec_item_idx_list.append(candidate_text_single)
                    matched_name = candidate_text_single
                    break
            if matched_name is None:
                continue

            candidate_pr = candidate_text.index(matched_name)
            scores[i, idxs[i,
                           candidate_pr]] = self.recall_budget - found_item_cnt
            found_item_cnt += 1
        return rec_item_idx_list

    def parsing_output_indices(self, scores, i, response_list, idxs,
                               candidate_text):
        rec_item_idx_list = []
        found_item_cnt = 0
        for j, item_detail in enumerate(response_list):
            if len(item_detail) < 1:
                continue

            if not item_detail.isdigit():
                continue

            pr = int(item_detail)
            if pr >= self.recall_budget:
                continue
            matched_name = candidate_text[pr]
            if matched_name in rec_item_idx_list:
                continue
            rec_item_idx_list.append(matched_name)
            scores[i, idxs[i, pr]] = self.recall_budget - found_item_cnt
            found_item_cnt += 1
            if len(rec_item_idx_list) >= self.recall_budget:
                break

        return rec_item_idx_list
