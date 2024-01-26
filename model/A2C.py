class ActorCritic(nn.Module):
    """An actor-critic network for parsing parameters.

    Using ``actor_critic.parameters()`` instead of set.union or list+list to avoid
    issue #449.

    :param nn.Module actor: the actor network.
    :param nn.Module critic: the critic network.
    """
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config[
            "embedding_size"]  # int type:the embedding size of lightGCN

        self.policy = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, 3),
            nn.Softmax(dim=-1)).to(config["device"])
        self.critic = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),
            nn.BatchNorm1d(self.latent_dim), nn.Linear(self.latent_dim,
                                                       1)).to(config["device"])
        self.gru = nn.GRU(config["embedding_size"],
                          config["embedding_size"],
                          num_layers=2)
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim * config["recall_budget"],
                      self.latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.latent_dim),
        ).to(config["device"])
        self.loss = nn.MSELoss()

    def sample_actions(self, policy):

        dist = Categorical(policy)
        action = dist.sample()  #可采取的action
        log_prob = dist.log_prob(action)  #每种action的概率

        return action.detach().numpy(), log_prob

    def actor(self, u_embedding):
        actions, log_prob = self.sample_actions(self.policy(u_embedding))
        return actions, log_prob
