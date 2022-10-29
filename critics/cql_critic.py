from infrastructure.base_class import BaseCritic
import torch
import torch.optim as optim
from torch import nn
from infrastructure import pytorch_util as ptu


class CQLCritic(BaseCritic):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.obs_dim = hparams['obs_dim']
        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['max_norm_clipping']
        self.gamma = hparams['gamma']
        self.optimizer_spec = hparams['critic_optim_spec']
        network_initializer = hparams['q_func']
        self.q_net = network_initializer()
        self.q_net_target = network_initializer()
        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.loss = nn.MSELoss()
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)
        self.cql_alpha = hparams['cql_alpha']

    def dqn_loss(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        qa_t_values = self.q_net(ob_no)
        q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
        qa_tp1_values = self.q_net_target(next_ob_no)
        if self.double_q:
            next_actions = self.q_net(next_ob_no).argmax(dim=1)
            q_tp1 = torch.gather(qa_tp1_values, 1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            q_tp1, _ = qa_tp1_values.max(dim=1)
        target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
        target = target.detach()
        loss = self.loss(q_t_values, target)
        return loss, qa_t_values, q_t_values

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)
        loss, qa_t_values, q_t_values = self.dqn_loss(
            ob_no, ac_na, next_ob_no, reward_n, terminal_n
        )
        num_path = torch.sum(terminal_n==1).item()+1
        sum_of_path_lengths = reward_n.shape[0]
        q_t_logsumexp = torch.log(torch.sum(torch.exp(qa_t_values), dim=1))
        cql_loss = self.cql_alpha*(q_t_logsumexp - q_t_values).sum()/sum_of_path_lengths + loss.mean()
        self.optimizer.zero_grad()
        cql_loss.backward()
        self.optimizer.step()
        info = {'Training Loss': ptu.to_numpy(loss), 'CQL Loss': ptu.to_numpy(cql_loss),
                'Data q-values': ptu.to_numpy(q_t_values).mean(), 'OOD q-values': ptu.to_numpy(q_t_logsumexp).mean()}
        return info

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs, **kwargs):
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
