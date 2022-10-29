from critics.dqn_critic import DQNCritic
from infrastructure.utils import normalize, FlexibleReplayBuffer
from infrastructure import pytorch_util as ptu
from policies import MLPPolicyAWAC, ArgmaxPolicy
from models.rnd_model import RNDModel
from agents.dqn_agent import DQNAgent
import numpy as np
import torch


class AWACAgent(DQNAgent):
    def __init__(self, env, agent_params, normalize_rnd=True, rnd_gamma=0.99):
        super(AWACAgent, self).__init__(env, agent_params)

        self.replay_buffer = FlexibleReplayBuffer(agent_params['batch_size'], 1)
        self.num_exploration_steps = agent_params['num_exploration_steps']
        self.offline_exploitation = agent_params['offline_exploitation']

        self.exploitation_critic = DQNCritic(agent_params)
        self.exploration_critic = DQNCritic(agent_params)

        self.exploration_model = RNDModel(agent_params)
        self.explore_weight_schedule = agent_params['explore_weight_schedule']
        self.exploit_weight_schedule = agent_params['exploit_weight_schedule']

        self.use_boltzmann = agent_params['use_boltzmann']
        self.actor = ArgmaxPolicy(self.exploitation_critic)
        self.eval_policy = self.awac_actor = MLPPolicyAWAC(
            self.agent_params['ac_dim'],
            self.agent_params['mean_net'],
            self.agent_params['logits_na'],
            self.agent_params['max_norm_clipping'],
            self.agent_params['awac_lambda'],
            self.agent_params['discrete'],
        )

        self.exploit_rew_shift = agent_params['exploit_rew_shift']
        self.exploit_rew_scale = agent_params['exploit_rew_scale']
        self.eps = agent_params['eps']

        self.running_rnd_rew_std = 1
        self.normalize_rnd = normalize_rnd
        self.rnd_gamma = rnd_gamma

    def get_qvals(self, critic, obs, action):
        qa_value = critic.q_net(obs)
        q_value = torch.gather(qa_value, 1, action.unsqueeze(1)).squeeze(1)
        return q_value.detach()

    def estimate_advantage(self, ob_no, ac_na, re_n, next_ob_no, terminal_n, n_actions=10):
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        re_n = ptu.from_numpy(re_n)
        next_ob_no = ptu.from_numpy(next_ob_no)
        terminal_n = ptu.from_numpy(terminal_n)
        dist_1 = self.awac_actor(ob_no)
        ac_prob = dist_1.probs
        if self.agent_params['discrete']:
            qa_val_s = self.actor.critic.q_net(ob_no)
        else:
            for _ in range(n_actions):
                pass
            qa_val_s = None
        v_pi = torch.sum(torch.multiply(ac_prob, qa_val_s), dim=1)
        q_vals = self.get_qvals(self.actor.critic, ob_no, ac_na)
        return q_vals - v_pi

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if self.t > self.num_exploration_steps:
            self.actor.set_critic(self.exploitation_critic)
            self.actor.use_boltzmann = False
        if (self.t > self.learning_start
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):
            exploit_weight = self.exploit_weight_schedule.value(self.t)
            explore_weight = self.explore_weight_schedule.value(self.t)
            expl_bonus = self.exploration_model.get_prediction(next_ob_no)
            expl_bonus = normalize(expl_bonus, np.mean(expl_bonus), np.std(expl_bonus))
            mixed_reward = explore_weight * expl_bonus + exploit_weight * re_n
            env_reward = (re_n + self.exploit_rew_shift) * self.exploit_rew_scale
            expl_model_loss = self.exploration_model.update(next_ob_no)
            exploration_critic_loss = self.exploration_critic.update(ob_no, ac_na, next_ob_no, mixed_reward, terminal_n)
            exploitation_critic_loss = self.exploitation_critic.update(ob_no, ac_na, next_ob_no, env_reward, terminal_n)
            adv_n = self.estimate_advantage(ob_no, ac_na, re_n, next_ob_no, terminal_n)
            actor_loss = self.awac_actor.update(ob_no, ac_na, adv_n)
            if self.num_param_updates % self.target_update_freq == 0:
                self.exploitation_critic.update_target_network()
                self.exploration_critic.update_target_network()
            log['Exploration Critic Loss'] = exploration_critic_loss['Training Loss']
            log['Exploitation Critic Loss'] = exploitation_critic_loss['Training Loss']
            log['Exploration Model Loss'] = expl_model_loss
            log['Actor Loss'] = actor_loss
            self.num_param_updates += 1
        self.t += 1
        return log

    def step_env(self):
        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)
        perform_random_action = np.random.random() < self.eps or self.t < self.learning_start
        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            processed = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(processed)
        next_obs, reward, done, info = self.env.step(action)
        self.last_obs = next_obs.copy()
        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)
        if done:
            self.last_obs = self.env.reset()
