import os.path as osp
import torch
import openai
import time
import asyncio
import numpy as np
from tqdm import tqdm
from recbole.model.abstract_recommender import SequentialRecommender

from utils import dispatch_openai_requests, dispatch_single_openai_requests


class Rl_1(SequentialRecommender):

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

            prompts_.append(prompt_)
            self.prompt_list.append([{'role': 'user', 'content': prompt}])
        return prompts_

    def con_prompt(self, dataset_name, i, interaction, idxs, actions,
                   actions_2, actions_3, actions_4):

        if dataset_name == 'ml-1m':

            for j in range(actions.size(1)):

                self.ini_len_l[i] = self.ini_len_l[i] + int(actions[i][j])
            user_his_text, candidate_text, candidate_text_order, candidate_idx = self.get_batch_inputs(
                interaction, idxs, i, 5)
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
                f"Please only output the {self.recall_budget} order numbers. Split these order numbers with line break.\n",
                f"Please only output the {self.recall_budget} order numbers and ignore any unnecessary steps. Split these order numbers with line break.",
                f"Attention! Just output the {self.recall_budget} order numbers and you don't need a lot of text. Split these order numbers with line break.",
                f"Please only output the {self.recall_budget} order numbers. Do not explain the reason or include any other words. Split these order numbers with line break.",
                f"Please only output the {self.recall_budget} order numbers. Split these order numbers with line break. Your output format should be like: \nx\n. x is the order number."
            ]

            prompt_4 = prompt_4_[actions_4[i][-1]]
            prompt_0 = prompt_0_[actions_2[i][-1]]
            prompt_3 = prompt_3_[actions_3[i][-1]]
            prompt = prompt_0 + prompt_1 + prompt_2 + prompt_3_[3] + prompt_4

            prompt_ = prompt_0 + prompt_1 + prompt_3

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
                f"Please only output the {self.recall_budget} order numbers. Split these order numbers with line break.\n",
                f"Please only output the {self.recall_budget} order numbers and ignore any unnecessary steps. Split these order numbers with line break.",
                f"Attention! Just output tthe {self.recall_budget} order numbers and you don't need a lot of text. Split these order numbers with line break.",
                f"Please only output the {self.recall_budget} order numbers. Do not explain the reason or include any other words. Split these order numbers with line break.",
                f"Please only output the {self.recall_budget} order numbers. Split these order numbers with line break. Your output format should be like: \nx\n. x is the order number."
            ]

            prompt_4 = prompt_4_[actions_4[i][-1]]
            prompt_0 = prompt_0_[actions_2[i][-1]]
            prompt_3 = prompt_3_[actions_3[i][-1]]
            prompt = prompt_0_[0] + prompt_1 + prompt_2 + prompt_3_[
                3] + prompt_4

            prompt_ = prompt_0 + prompt_1 + prompt_3 + prompt_4

        elif dataset_name == 'Games':
            for j in range(actions.size(1)):

                self.ini_len_l[i] = self.ini_len_l[i] + int(actions[i][j])
            user_his_text, candidate_text, candidate_text_order, candidate_idx = self.get_batch_inputs(
                interaction, idxs, i, self.ini_len_l[i])

            prompt_0_ = [
                "", "You are a game expert.\n",
                "You are a game recommender.\n",
                "You are a game ranker to match user's interest.\n",
                "You are familiar with games and good at catching people's gaming interest.\n"
            ]
            prompt_1 = f"I've purchased the following games in the past in order:\n{user_his_text}\n\n"
            prompt_2 = f"Now there are {self.recall_budget} candidate games that I can purchase next:\n{candidate_text_order}\n"
            prompt_3_ = [
                f"Please rank these {self.recall_budget} games in order of priority from highest to lowest.\n",
                f"Please rank these {self.recall_budget} games according to my purchasing records.\n",
                f"Please rank these {self.recall_budget} games. When ranking, only consider my gaming preferences according to my purchaing records.\n",
                f"Please rank these {self.recall_budget} games by measuring the possibilities that I would like to purchase next most, according to my purchasing records. Please think step by step.\n",
                f"Please rank these {self.recall_budget} games based on my gaming preferences which are inferred from my purchasing records.\n",
                f"Please rank these {self.recall_budget} games after updating my gaming preferences according to my purchasing records.\n",
                f"Please rank these {self.recall_budget} games after calculating the similarity between them and my gaming preferences, according to my purchaing records.\n",
                f"Please rank these {self.recall_budget} games according to the similarity score from high to low after computing the similarity between each candidate product and the whole purchasing records.\n",
                f"Please rank these {self.recall_budget} games and refine the ranking, according to the purchasing records.\n"
            ]
            prompt_4_ = [
                f"Please only output the {self.recall_budget} order numbers. Split these order numbers with line break.\n",
                f"Please only output the {self.recall_budget} order numbers and ignore any unnecessary steps. Split these order numbers with line break.",
                f"Attention! Just output the {self.recall_budget} order numbers and you don't need a lot of text. Split these order numbers with line break.",
                f"Please only output the {self.recall_budget} order numbers. Do not explain the reason or include any other words. Split these order numbers with line break.",
                f"Please only output the {self.recall_budget} order numbers. Split these order numbers with line break. Your output format should be like: x\n. x is the order number."
            ]

            prompt_4 = prompt_4_[actions_4[i][-1]]
            prompt_0 = prompt_0_[actions_2[i][-1]]
            prompt_3 = prompt_3_[actions_3[i][-1]]
            prompt = prompt_0 + prompt_1 + prompt_2 + prompt_3 + prompt_4
            prompt_ = prompt_0 + prompt_1 + prompt_3

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

        openai_responses = self.dispatch_openai_api_requests(
            self.prompt_list, batch_size)

        scores = torch.full((idxs.shape[0], self.n_items), -10000.)
        for i, openai_response in enumerate(tqdm(openai_responses)):
            user_his_text, candidate_text, candidate_text_order, candidate_idx = self.get_batch_inputs(
                interaction, idxs, i, self.ini_len_l[i])

            response = openai_response['choices'][0]['message']['content']
            response_list = response.split('\n')

            if self.dataset_name == 'ml-1m':
                rec_item_idx_list, prs = self.parsing_output_indices(
                    scores, i, response_list, idxs, candidate_text)
                if prs_ == None:
                    prs_ = prs
                else:
                    prs_ = torch.concat((prs_, prs), 0)

            elif self.dataset_name == 'lastfm':
                rec_item_idx_list, prs = self.parsing_output_indices(
                    scores, i, response_list, idxs, candidate_text)
                if prs_ == None:
                    prs_ = prs
                else:
                    prs_ = torch.concat((prs_, prs), 0)

            else:
                rec_item_idx_list, prs = self.parsing_output_indices(
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
        if len(prs) < 20:
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
        if len(prs) < 20:
            prs = []
            for j in range(idxs.shape[1]):
                prs.append(idxs[i, j])
        return rec_item_idx_list, torch.tensor(prs).reshape(1, idxs.shape[1])
