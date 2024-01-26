import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.data import create_dataset
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.utils import init_seed, init_logger, get_trainer, set_color
from utils import get_model
from model.backbone import LightGCN, ActorCritic,Rank_state
from trainer_te import SelectedUserTrainer
import torch.optim as optim
import torch.nn as nn
import os
import torch
import numpy as np
from recbole.utils import EvaluatorType, set_color
from recbole.data.interaction import Interaction

# os.environ["CUDA_VISIBLE_DEVICES"] = '3' 
# torch.set_default_dtype(torch.float32)
def evaluate(model_name, dataset_name,last_episode_model, pretrained_file, **kwargs):
    # configurations initialization
    props = [
        'props/Rank.yaml', f'props/{dataset_name}.yaml', 'openai_api.yaml',
        'props/overall.yaml', 'props/lightGCN.yaml'
    ]
    print(props)
    model_class = get_model(model_name)

    # configurations initialization
    config = Config(model=model_class,
                    dataset=dataset_name,
                    config_file_list=props,
                    config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = SequentialDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = model_class(config, train_data.dataset).to(config['device'])
    model_backbone = LightGCN(config, train_data._dataset).to(config["device"])
    a2c = ActorCritic(config,3).to(config['device'])
    a2c_2 = ActorCritic(config,5).to(config['device'])
    a2c_3 = ActorCritic(config,9).to(config['device'])
    a2c_4 = ActorCritic(config,5).to(config['device'])
    rank_state=Rank_state(config["embedding_size"]).to(config['device'])
    # 默认方法

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file,
                                map_location=torch.device('cuda'))
        logger.info(f'Loading from {pretrained_file}')
        model_backbone.load_state_dict(checkpoint['state_dict'], strict=False)
        model_backbone.load_other_parameter(checkpoint.get("other_parameter"))
    if last_episode_model != '':
        
        a2c = torch.load("saved/ml-1m_last_a2c_400.pth",
                         map_location=torch.device('cuda'))
        a2c_2 = torch.load("saved/ml-1m_last_a2c_2_400.pth",
                            map_location=torch.device('cuda'))
        a2c_3 = torch.load("saved/ml-1m_last_a2c_3_400.pth",
                            map_location=torch.device('cuda'))
        a2c_4 = torch.load("saved/ml-1m_last_a2c_4_400.pth",
                            map_location=torch.device('cuda'))

    logger.info(model)
    selected_uids, sampled_items = load_selected_users(config, dataset)
    # trainer loading and initialization
    trainer = SelectedUserTrainer(config, model, dataset)


    model_backbone.train()
    with torch.no_grad():
        user_all_embeddings, item_all_embeddings = model_backbone.forward()
    user_embedding = user_all_embeddings[selected_uids].float()
    if config["boots"]:
        user_embedding = torch.tensor(
            np.tile(user_embedding.cpu(), [config["boots"], 1])).to(config['device']).float()
    prompt_embedding = torch.zeros(user_embedding.shape[0], 64).to(config['device']).float()
    rank_embedding = torch.zeros(user_embedding.shape[0], 64).to(config['device']).float()
    actions = None
    actions_2 = None
    actions_3 = None
    actions_4 = None
    k=0
    reward=0
    results={"ndcg@1":0.55,"ndcg@5":0.55,"ndcg@10":0.55,"ndcg@20":0.55} 
    
    for r in range(15):
        
    #   if not done:
        # reward_p=torch.tensor(reward_test["ndcg@20"]).detach().float()
        a2c.train()
        a2c_2.train()
        a2c_3.train()
        a2c_4.train()
        # start = time.time()
        value_s = a2c.critic(prompt_embedding+rank_embedding.detach() 
                                )
        value_s_2 = a2c_2.critic(prompt_embedding+rank_embedding.detach()
                                )
        value_s_3 = a2c_3.critic(prompt_embedding+rank_embedding.detach())
        value_s_4 = a2c_4.critic(prompt_embedding+rank_embedding.detach())

        if r == 0:
            action, log_prob = a2c.actor(user_embedding+rank_embedding.detach())
            action_2, log_prob_2 = a2c_2.actor(user_embedding+rank_embedding.detach())
            action_3, log_prob_3 = a2c_3.actor(user_embedding+rank_embedding.detach())
            action_4, log_prob_4 = a2c_4.actor(user_embedding+rank_embedding.detach())
            # if config["boots"]:
            #     action = np.tile(action, config["boots"])
            action = torch.tensor(action).reshape(action.shape[0], 1).cpu()
            actions = torch.concat([action]).cpu()
            action_2 = torch.tensor(action_2).reshape(action_2.shape[0], 1).cpu()
            actions_2 = torch.concat([action_2]).cpu()
            action_3 = torch.tensor(action_3).reshape(action_3.shape[0], 1).cpu()
            actions_3 = torch.concat([action_3]).cpu()
            action_4 = torch.tensor(action_4).reshape(action_4.shape[0], 1).cpu()
            actions_4 = torch.concat([action_4]).cpu()
            # if episode==0:
            prompt_embedding = trainer.prompts(
            test_data,
            actions,
            actions_2,actions_3,actions_4,
            load_best_model=False,
            show_progress=config['show_progress']).to(config['device']).float()
            reward_test, prs = trainer.evaluate()
            # else:
            #     prompt_embedding = trainer.prompts_(
            #     test_data,
            #     actions,
            #     actions_2,
            #     actions_3,
            #     actions_4,
            #     load_best_model=False,
            #     show_progress=config['show_progress']).to(config['device']).float()
            #     reward_test, prs = trainer.evaluate_()
            
            # reward_test={"ndcg@1":0.1,"ndcg@5":0.2,"ndcg@10":0.3,"ndcg@20":0.4}
        else:
            action, log_prob = a2c.actor(prompt_embedding+rank_embedding.detach()
                                            )
            action_2, log_prob_2 = a2c_2.actor(prompt_embedding+rank_embedding.detach()
                                            )
            action_3, log_prob_3 = a2c_3.actor(prompt_embedding+rank_embedding.detach()
                                            )
            action_4, log_prob_4 = a2c_4.actor(prompt_embedding+rank_embedding.detach()
                                            )
            action = torch.tensor(action).reshape(action.shape[0], 1).cpu()
            action_2 = torch.tensor(action_2).reshape(action_2.shape[0], 1).cpu()
            action_3 = torch.tensor(action_3).reshape(action_3.shape[0], 1).cpu()
            action_4 = torch.tensor(action_4).reshape(action_4.shape[0], 1).cpu()
        
            actions = torch.concat((actions.cpu(), action), 1)               
            actions_2 = torch.concat((actions_2.cpu(), action_2), 1)
            actions_3 = torch.concat((actions_3.cpu(), action_3), 1)
            actions_4 = torch.concat((actions_4.cpu(), action_4), 1)
            prompt_embedding = trainer.prompts_(
                test_data,
                actions,
                actions_2,
                actions_3,actions_4,
                load_best_model=False,
                show_progress=config['show_progress']).to(config['device']).float()
            
            reward_test, prs = trainer.evaluate_()
            # reward_test={"ndcg@1":0.1,"ndcg@5":0.2,"ndcg@10":0.3,"ndcg@20":0.4}
        # reward_test={"ndcg@1":0.1,"ndcg@5":0.2,"ndcg@10":0.3,"ndcg@20":0.4}
        # prs =
        reward = torch.tensor(reward_test["ndcg@10"]).detach().float()
        # logger.info(set_color('test result', 'yellow') + f': {reward_test}')
        item_embedding = torch.zeros(
        action.shape[0],
        config["recall_budget"] * config["embedding_size"])
        for i in range(action.shape[0]):
            item_embedding_ = item_all_embeddings[prs[i].long()]
            item_embedding[i] = item_embedding_.reshape(
                1, config["recall_budget"] * config["embedding_size"])

        hidden = torch.zeros(1, action.shape[0],
                            config["embedding_size"]).to(config['device']).float()

        output, hidden = rank_state.rank_(
            item_embedding.reshape(config["recall_budget"], action.shape[0],
                                config["embedding_size"]).float().to(config['device']), hidden.float().to(config['device']))
        # rank_embedding = a2c.mlp(
        #     output.reshape(
        #         prompt_embedding.size(0),
        #         config["recall_budget"] * config["embedding_size"]))
        rank_embedding = hidden.reshape(action.shape[0],
                                        config["embedding_size"])
            

        logger.info(set_color('test result', 'yellow') + f': {reward_test}')
        # if reward_test["ndcg@10"] > results["ndcg@10"]:
        #     results = reward_test
            
        #     k=0
        #     torch.save(a2c, "./saved/" + dataset_name + "_last_a2c.pth")
        #     torch.save(a2c_2, "./saved/" + dataset_name+"_last_a2c_2.pth")
        #     torch.save(a2c_3, "./saved/" + dataset_name +"_last_a2c_3.pth")
        #     torch.save(a2c_4, "./saved/" + dataset_name +"_last_a2c_4.pth")
        
        # k=k+1
        # if k>7 or r==14:
                
        #         break
        
   

    # return config['model'], config['dataset'], {
    #     'valid_score_bigger': config['valid_metric_bigger'],
    #     'test_result': reward_test
    # }


def load_selected_users(config, dataset):
    selected_users = []
    sampled_items = []
    selected_user_file = os.path.join(
        config['data_path'],
        f'{config["dataset"]}.{config["selected_user_suffix"]}')
    user_token2id = dataset.field2token_id['user_id']
    item_token2id = dataset.field2token_id['item_id']
    with open(selected_user_file, 'r', encoding='utf-8') as file:
        for line in file:
            uid, iid_list = line.strip().split('\t')
            selected_users.append(uid)
            sampled_items.append([
                item_token2id[_] if (_ in item_token2id) else 0
                for _ in iid_list.split(' ')
            ])
    selected_uids = list([user_token2id[_] for _ in selected_users])
    return selected_uids, sampled_items
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, default="Rl_1", help="model name")
    parser.add_argument('-d', type=str, default='ml-1m', help='dataset name')
    parser.add_argument('-p',
                        type=str,
                        default='saved/pretrained-ml-1m.pth',
                        help='pre-trained model path')
    parser.add_argument('-l',
                        type=str,
                        default='ml-1m_last_a2c_2',
                        help='last episode model path')
    args, unparsed = parser.parse_known_args()
    print(args)

    evaluate(args.m, args.d, args.l, pretrained_file=args.p)
