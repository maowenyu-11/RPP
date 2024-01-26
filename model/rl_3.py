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


class Rl_3(SequentialRecommender):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.max_tokens = config['max_tokens']

        self.ini_len = config['ini_len']
        self.recall_budget = config['recall_budget']
        self.boots = config['boots']
        self.data_path = config['data_path']
        self.dataset_name = dataset.dataset_name
        self.id_token = dataset.field2id_token['item_id']
        self.item_text = self.load_text()
        self.logger.info(
            f'Avg. t = {np.mean([len(_) for _ in self.item_text])}')

        self.fake_fn = torch.nn.Linear(1, 1)
        self.prompt_list = []
        self.ini_len_l = []

    def load_model(self):

        base_model = "llama-2-hf-7b"  # decapoda-research/llama-13b-hf
        lora_weights = "lora-alpaca"  # chansung/gpt4-alpaca-lora-13b、tloen/alpaca-lora-7b
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

    def evaluate(
        self,
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
            generation_output = self.model.generate(
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
        response = self.get_response(output)
        return response

    def generate_prompt(self, instruction, input=None):
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
                    item_id, name, url, picture_url = line.strip().split('\t')
                    token_text[item_id] = name
            for i, token in enumerate(self.id_token):
                # print(self.id_token)
                if token == '[PAD]': continue
                raw_text = token_text[token]
                item_text.append(raw_text)
            return item_text

        elif self.dataset_name == 'book-crossing':
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, book_title, book_author, publication_year, publisher = line.strip(
                    ).split('\t')
                    token_text[item_id] = book_title
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

    def prompt_con(self, interaction, idxs, actions, actions_2, actions_3,
                   actions_4):
        origin_batch_size = idxs.shape[0]
        if self.boots:
            """ 
            bootstrapping is adopted to alleviate position bias
            `fix_enc` is invalid in this case"""
            idxs = np.tile(idxs, [self.boots, 1])
            np.random.shuffle(idxs.T)
        batch_size = idxs.shape[0]
        pos_items = interaction[self.POS_ITEM_ID]
        # prompts = []
        prompts_ = []
        self.prompt_list = []
        self.ini_len_l = [self.ini_len] * idxs.shape[0]

        for i in tqdm(range(batch_size)):

            prompt, prompt_ = self.con_prompt(self.dataset_name, i,
                                              interaction, idxs, actions,
                                              actions_2, actions_3, actions_4)

            # prompts.append(prompt)
            prompts_.append(prompt_)
            self.prompt_list.append(prompt)
        return prompts_

    def con_prompt(self, dataset_name, i, interaction, idxs, actions,
                   actions_2, actions_3, actions_4):

        if dataset_name == 'ml-1m':

            for j in range(actions.size(1)):

                self.ini_len_l[i] = self.ini_len_l[i] + int(actions[i][j])
            user_his_text, candidate_text, candidate_text_order, candidate_idx = self.get_batch_inputs(
                interaction, idxs, i, self.ini_len_l[i])
            prompt_0_ = [
                "", "You are a movie expert.\n",
                "You are a movie recommender\n",
                "You are a movie ranker to match user's interest.\n",
                "You are familiar with movies and good at catching people's movie interest.\n"
            ]
            prompt_1 = f"I've watched the following movies in the past in order:\n{user_his_text}\n\n"
            prompt_2 = f"Now there are {self.recall_budget} candidate movies that I can watch next:\n{candidate_text_order}\n"
            prompt_3_ = [
                f"Please rank these {self.recall_budget} movies in order of priority from highest to lowest.\n",
                f"Please rank these {self.recall_budget} movies according to my watching history.\n",
                f"Please rank these {self.recall_budget} movies. When ranking, only consider my film preferences according to my watching history.\n",
                f"Please rank these {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history. Please think step by step.\n",
                f"Please rank these {self.recall_budget} movies based on my film preferences which are inferred from my watching history.\n",
                f"Please rank these {self.recall_budget} movies after updating my film preferences according to my watching history.\n",
                f"Please rank these {self.recall_budget} movies after calculating the similarity between them and my movie preferences, according to my watching history.\n",
                f"Please rank these {self.recall_budget} movies according to the similarity score from high to low after computing the similarity between each candidate movie and the whole watching history.\n",
                f"Please rank these {self.recall_budget} movies and refine the ranking, according to the watching history.\n"
            ]
            prompt_4_ = [
                f"Please only output the {self.recall_budget} ranking results with order numbers. Split these order numbers with line break.\n",
                f"Please only output the {self.recall_budget} ranking results with order numbers and ignore any unnecessary steps. Split these order numbers with line break.",
                f"Attention! Just output the {self.recall_budget} ranking results with order numbers and you don't need a lot of text. Split these order numbers with line break.",
                f"Please only output the {self.recall_budget} ranking results with order numbers. Do not explain the reason or include any other words. Split these order numbers with line break.",
                f"Please only output the {self.recall_budget} ranking results with order numbers. Split these order numbers with line break. Your output format should be like: \nx.movie name\n. x is the order number."
            ]

            prompt_4 = prompt_4_[actions_4[i][-1]]
            prompt_0 = prompt_0_[actions_2[i][-1]]
            prompt_3 = prompt_3_[actions_3[i][-1]]
            prompt = prompt_0 + prompt_1 + prompt_2 + prompt_3 + prompt_4

            prompt_ = prompt_0 + prompt_1 + prompt_3 + prompt_4
            # recent_item = user_his_text[-1][user_his_text[-1].find('. ') + 2:]
            # prompt=f"I've watched the following movies in the past in order:\n{user_his_text}\n\n" \
            #         f"Now there are {self.recall_budget} candidate movies that I can watch next:\n{candidate_text_order}\n" \
            #         f"Please rank these {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history. Please think step by step.\n" \
            #         f"Note that my most recently watched movie is {recent_item}. " \
            #         f"Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."
            # prompt_=f"I've watched the following movies in the past in order:\n{user_his_text}\n\n" \
            #         f"Please rank these {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history. Please think step by step.\n" \
            #         f"Note that my most recently watched movie is {recent_item}. " \
            #         f"Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."

        elif dataset_name == 'lastfm':

            for j in range(actions.size(1)):

                self.ini_len_l[i] = self.ini_len_l[i] + int(actions[i][j])
            user_his_text, candidate_text, candidate_text_order, candidate_idx = self.get_batch_inputs(
                interaction, idxs, i, self.ini_len_l[i])
            prompt_0_ = [
                "", "You are a music expert.\n",
                "You are a music recommender\n",
                "You can rank musics to match user's interest.\n",
                "You are familiar with musics and good at catching people's music preference.\n"
            ]
            prompt_1 = f"I've listened the following musics in the past in order:\n{user_his_text}\n\n"
            prompt_2 = f"Now there are {self.recall_budget} candidate musics that I can listen next:\n{candidate_text_order}\n"
            prompt_3_ = [
                f"Please rank these {self.recall_budget} musics in order of priority from highest to lowest.\n",
                f"Please rank these {self.recall_budget} musics according to my listening history.\n",
                f"Please rank these {self.recall_budget} musics. When ranking, only consider my muisc preferences according to my listeninging history.\n",
                f"Please rank these {self.recall_budget} musics by measuring the possibilities that I would like to listen next most, according to my listening history. Please think step by step.\n",
                f"Please rank these {self.recall_budget} musics based on my music preferences which are inferred from my listening history.\n",
                f"Please rank these {self.recall_budget} musics after updating my music preferences according to my listening history.\n",
                f"Please rank these {self.recall_budget} musics after calculating the similarity between them and my music preferences, according to my listening history.\n",
                f"Please rank these {self.recall_budget} musics according to the similarity score from high to low after computing the similarity between each candidate music and the whole listening history.\n",
                f"Please rank these {self.recall_budget} musics and refine the ranking, according to the listening history.\n"
            ]
            prompt_4_ = [
                f"Please only output the {self.recall_budget} ranking results with order numbers. Split these order numbers with line break.\n",
                f"Please only output the {self.recall_budget} ranking results with order numbers and ignore any unnecessary steps. Split these order numbers with line break.",
                f"Attention! Just output the {self.recall_budget} ranking results with order numbers and you don't need a lot of text. Split these order numbers with line break.",
                f"Please only output the {self.recall_budget} ranking results with order numbers. Do not explain the reason or include any other words. Split these order numbers with line break.",
                f"Please only output the {self.recall_budget} ranking results with order numbers. Split these order numbers with line break. Your output format should be like: \nx.music name\n. x is the order number."
            ]

            prompt_4 = prompt_4_[actions_4[i][-1]]
            prompt_0 = prompt_0_[actions_2[i][-1]]
            prompt_3 = prompt_3_[actions_3[i][-1]]
            prompt = prompt_0 + prompt_1 + prompt_2 + prompt_3 + prompt_4

            prompt_ = prompt_0 + prompt_1 + prompt_3 + prompt_4

        elif dataset_name == 'Games':
            for j in range(actions.size(1)):

                self.ini_len_l[i] = self.ini_len_l[i] + int(actions[i][j])
            user_his_text, candidate_text, candidate_text_order, candidate_idx = self.get_batch_inputs(
                interaction, idxs, i, self.ini_len_l[i])
            recent_item = user_his_text[-1][user_his_text[-1].find('. ') + 2:]

            prompt_0_ = [
                "", "You are a game expert.\n",
                "You are a game recommender.\n",
                "You can rank games to match user's interest.\n",
                "You are familiar with games and good at catching people's gaming preference.\n"
            ]
            prompt_1 = f"I've purchased the following products in the past in order:\n{user_his_text}\n\n"
            prompt_2 = f"Now there are {self.recall_budget} candidate products that I can purchase next:\n{candidate_text_order}\n"
            prompt_3_ = [
                f"Please rank these {self.recall_budget} products.\n",
                f"Please rank these {self.recall_budget} products according to my purchasing records.\n",
                f"Please rank these {self.recall_budget} products. When ranking, only consider my gaming preferences according to my purchaing records.\n",
                f"Please rank these {self.recall_budget} products by measuring the possibilities that I would like to purchase next most, according to my purchasing records. Please think step by step.\n",
                f"Please rank these {self.recall_budget} products based on my gaming preferences which are inferred from my purchasing records.\n",
                f"Please rank these {self.recall_budget} products after updating my gaming preferences according to my purchasing records.\n",
                f"Please rank these {self.recall_budget} products after calculating the similarity between them and my gaming preferences, according to my purchaing records.\n",
                f"Please rank these {self.recall_budget} products according to the similarity score from high to low after computing the similarity between each candidate product and the whole purchasing records.\n",
                f"Please rank these {self.recall_budget} products and refine the ranking, according to the purchasing records.\n"
            ]
            prompt_4_ = [
                f"Please only output the {self.recall_budget} ranking results with order numbers. Split these order numbers with line break.\n",
                f"Please only output the {self.recall_budget} ranking results with order numbers and ignore any unnecessary steps. Split these order numbers with line break.",
                f"Attention! Just output the {self.recall_budget} ranking results with order numbers and you don't need a lot of text. Split these order numbers with line break.",
                f"Please only output the {self.recall_budget} ranking results with order numbers. Do not explain the reason or include any other words. Split these order numbers with line break.",
                f"Please only output the {self.recall_budget} ranking results with order numbers. Split these order numbers with line break. Your output format should be like: \nx.product name\n. x is the order number."
            ]

            prompt_4 = prompt_4_[actions_4[i][-1]]
            prompt_0 = prompt_0_[actions_2[i][-1]]
            prompt_3 = prompt_3_[actions_3[i][-1]]
            prompt = prompt_0 + prompt_1 + prompt_2 + prompt_3 + prompt_4
            prompt_ = prompt_0 + prompt_1 + prompt_3 + prompt_4

            # prompt = f"I've purchased the following products in the past in order:\n{user_his_text}\n\n" \
            #         f"Now there are {self.recall_budget} candidate products that I can consider to purchase next:\n{candidate_text_order}\n" \
            #         f"Please rank these {self.recall_budget} products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Please think step by step.\n" \
            #         f"Note that my most recently purchased product is {recent_item}. " \
            #         f"Please only output the order numbers after ranking. Split these order numbers with line break."

            # prompt_= f"I've purchased the following products in the past in order:\n{user_his_text}\n\n" \
            #         f"Please rank these {self.recall_budget} products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Please think step by step.\n" \
            #         f"Note that my most recently purchased product is {recent_item}. " \
            #         f"Please only output the order numbers after ranking. Split these order numbers with line break."

        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        return prompt, prompt_

    def predict_on_subsets(self, interaction, idxs):
        """
        Main function to rank with LLMs

        :param interaction:
        :param idxs: item id retrieved by candidate generation models [batch_size, candidate_size]
        :return:
        """

        llama_responses = []
        origin_batch_size = idxs.shape[0]
        prs_ = None
        if self.boots:
            """ 
            bootstrapping is adopted to alleviate position bias
            `fix_enc` is invalid in this case"""
            idxs = np.tile(idxs, [self.boots, 1])
            np.random.shuffle(idxs.T)
        batch_size = idxs.shape[0]
        pos_items = interaction[self.POS_ITEM_ID]

        for i in range(batch_size):
            llama_responses.append(self.evaluate(self.prompt_list[i]))
            # print(llama_responses[i])

        scores = torch.full((idxs.shape[0], self.n_items), -10000.)
        print(scores.shape[0])
        for i, response in enumerate(tqdm(llama_responses)):
            user_his_text, candidate_text, candidate_text_order, candidate_idx = self.get_batch_inputs(
                interaction, idxs, i, self.ini_len_l[i])
            # try:
            # response = llama_response
            response_list = response.split('\n')

            # self.logger.info(self.prompt_list[i])
            # self.logger.info(response)
            # self.logger.info(f'Here are candidates: {candidate_text}')
            # self.logger.info(f'Here are answer: {response_list}')

            if self.dataset_name == 'ml-1m':
                rec_item_idx_list, prs = self.parsing_output_text(
                    scores, i, response_list, idxs, candidate_text)
                if prs_ == None:
                    prs_ = prs
                else:
                    prs_ = torch.concat((prs_, prs), 0)

            elif self.dataset_name == 'lastfm':
                rec_item_idx_list, prs = self.parsing_output_text(
                    scores, i, response_list, idxs, candidate_text)
                if prs_ == None:
                    prs_ = prs
                else:
                    prs_ = torch.concat((prs_, prs), 0)

            elif self.dataset_name == 'book-crossing':
                rec_item_idx_list, prs = self.parsing_output_text(
                    scores, i, response_list, idxs, candidate_text)
                if prs_ == None:
                    prs_ = prs
                else:
                    prs_ = torch.concat((prs_, prs), 0)

            else:
                rec_item_idx_list, prs = self.parsing_output_text(
                    scores, i, response_list, idxs, candidate_text)
                if prs_ == None:
                    prs_ = prs
                else:
                    prs_ = torch.concat((prs_, prs), 0)

            if int(pos_items[i % origin_batch_size]) in candidate_idx:
                target_text = candidate_text[candidate_idx.index(
                    int(pos_items[i % origin_batch_size]))]
                # try:
                #     ground_truth_pr = rec_item_idx_list.index(target_text)
                #     self.logger.info(
                #         f'Ground-truth [{target_text}]: Ranks {ground_truth_pr}'
                #     )
                # except:
                #     self.logger.info(f'Fail to find ground-truth items.')
                #     print(target_text)
                #     print(rec_item_idx_list)
                #     continue
        if self.boots:
            scores = scores.view(self.boots, -1, scores.size(-1))
            scores = scores.sum(0)
        return scores, prs_

    def get_batch_inputs(self, interaction, idxs, i, ini_len):
        user_his = interaction[self.ITEM_SEQ]
        user_his_len = interaction[self.ITEM_SEQ_LEN]
        origin_batch_size = user_his.size(0)
        real_his_len = min(ini_len, user_his_len[i % origin_batch_size].item())
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

    def parsing_output_text(self, scores, i, response_list, idxs,
                            candidate_text):
        rec_item_idx_list = []
        prs = []
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
            prs.append(idxs[i, candidate_pr])
            found_item_cnt += 1
        if len(prs) < self.recall_budget:
            prs = []
            for j in range(idxs.shape[1]):
                prs.append(idxs[i, j])
        return rec_item_idx_list, torch.tensor(prs).reshape(1, idxs.shape[1])

    def parsing_output_indices(self, scores, i, response_list, idxs,
                               candidate_text):
        rec_item_idx_list = []
        prs = []
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
            prs.append(idxs[i, pr])
            found_item_cnt += 1
            if len(rec_item_idx_list) >= self.recall_budget:
                break
        if len(prs) < self.recall_budget:
            prs = []
            for j in range(idxs.shape[1]):
                prs.append(idxs[i, j])
        return rec_item_idx_list, torch.tensor(prs).reshape(1, idxs.shape[1])
