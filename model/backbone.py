import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from torch.distributions import Categorical


class LightGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(
            np.float32)

        # load parameters info
        self.latent_dim = config[
            "embedding_size"]  # int type:the embedding size of lightGCN
        self.n_layers = config[
            "n_layers"]  # int type:the layer num of lightGCN
        # self.reg_weight = config[
        #     "reg_weight"]  # float32 type: the weight decay for l2 normalization
        # self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users,
                                                 embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items,
                                                 embedding_dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users),
                [1] * inter_M.nnz))
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix,
                                             all_embeddings.float())
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings


# class Policy(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.latent_dim = config[
#             "embedding_size"]  # int type:the embedding size of lightGCN
#         self.mlp = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim),
#                                  nn.ReLU(), nn.Linear(self.latent_dim, 5),
#                                  nn.Softmax(dim=-1)).to(config["device"])

#     def forward(self,u):
#         a = self.mlp(u).max(1)[1]
#         return a
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data,
                                              a=0,
                                              mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'default':
                pass
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d'
        ) != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class ActorCritic(nn.Module):
    """An actor-critic network for parsing parameters.

    Using ``actor_critic.parameters()`` instead of set.union or list+list to avoid
    issue #449.

    :param nn.Module actor: the actor network.
    :param nn.Module critic: the critic network.
    """
    def __init__(self, config,n_actions):
        super().__init__()
        self.latent_dim = config[
            "embedding_size"]  # int type:the embedding size of lightGCN

        self.actor_network = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, n_actions),
            nn.Softmax(dim=-1)).to(config["device"])
        self.critic = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),
            nn.BatchNorm1d(self.latent_dim), nn.Linear(self.latent_dim,
                                                       1)).to(config["device"])
        
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.latent_dim * config["recall_budget"],
        #               self.latent_dim),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(self.latent_dim),
        # ).to(config["device"])
        self.loss = nn.MSELoss()

    def sample_actions(self, policy):

        dist = Categorical(policy)
        action = dist.sample()  #可采取的action
        log_prob = dist.log_prob(action)  #每种action的概率

        return action.detach().cpu().numpy(), log_prob.cuda()

    def actor(self, state):
        actions, log_prob = self.sample_actions(self.actor_network(state))
        return actions, log_prob
    
    
class Rank_state(nn.Module):
    """An actor-critic network for parsing parameters.

    Using ``actor_critic.parameters()`` instead of set.union or list+list to avoid
    issue #449.

    :param nn.Module actor: the actor network.
    :param nn.Module critic: the critic network.
    """
    def __init__(self, embedding_size):
        super().__init__()
        # self.latent_dim = config[
        #     "embedding_size"]  # int type:the embedding size of lightGCN

        self.rank_ = nn.GRU(embedding_size,
                          embedding_size,
                          num_layers=1)