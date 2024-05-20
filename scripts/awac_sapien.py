import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple
from config import Args
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from buffer import Demos_ReplayBuffer as ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import wandb

# ManiSkill specific imports
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset

TensorBatch = List[torch.Tensor]

# Define actor and critic architectures (fill with appropriate layers and dimensions)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Hyperparameters for action distribution
        self._min_log_std = -5.0
        self._max_log_std = 2.0
        self._min_action = torch.tensor(env.unwrapped.single_action_space.low).to("cuda")
        self._max_action = torch.tensor(env.unwrapped.single_action_space.high).to("cuda")

        # Network architecture
        self.fc1 = nn.Sequential(
            layer_init(nn.Linear(np.array(env.unwrapped.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.unwrapped.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.unwrapped.single_action_space.shape))
        
    def _get_policy(self, state: torch.Tensor) -> torch.distributions.Distribution:
        x = self.fc1(state)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self._min_log_std  + 0.5 * (self._max_log_std  - self._min_log_std ) * (log_std + 1)  # From SpinUp / Denis Yarats
        return torch.distributions.Normal(mean, log_std.exp())

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        policy = self._get_policy(state)
        return policy.log_prob(action).sum(-1, keepdim=True)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(self._min_action, self._max_action)
        log_prob = self.log_prob(state, action)
        return action, log_prob

    def act(self, state: torch.Tensor) -> torch.Tensor:
        policy = self._get_policy(state)
        return policy.sample() if self.fc1.training else policy.mean


class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        state_dim = np.array(env.unwrapped.single_observation_space.shape).prod()
        action_dim = np.prod(env.unwrapped.single_action_space.shape)
        hidden_dim = 256  # Define the size of hidden layers
        
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_value = self._mlp(torch.cat([state, action], dim=-1))
        return q_value


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class AdvantageWeightedActorCritic:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 5e-3,  # parameter for the soft target update,
        awac_lambda: float = 1.0,
        exp_adv_max: float = 100.0,
    ):
        self._actor = actor
        self._actor_optimizer = actor_optimizer

        self._critic_1 = critic_1
        self._critic_1_optimizer = critic_1_optimizer
        self._target_critic_1 = deepcopy(critic_1)

        self._critic_2 = critic_2
        self._critic_2_optimizer = critic_2_optimizer
        self._target_critic_2 = deepcopy(critic_2)

        self._gamma = gamma
        self._tau = tau
        self._awac_lambda = awac_lambda
        self._exp_adv_max = exp_adv_max

    def _actor_loss(self, states, actions):
        with torch.no_grad():
            pi_action, _ = self._actor(states)
            v = torch.min(
                self._critic_1(states, pi_action), self._critic_2(states, pi_action)
            )

            q = torch.min(
                self._critic_1(states, actions), self._critic_2(states, actions)
            )
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self._awac_lambda), self._exp_adv_max
            )

        action_log_prob = self._actor.log_prob(states, actions)
        loss = (-action_log_prob * weights).mean()
        return loss

    def _critic_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_actions, _ = self._actor(next_states)

            q_next = torch.min(
                self._target_critic_1(next_states, next_actions),
                self._target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self._gamma * (1.0 - dones) * q_next

        q1 = self._critic_1(states, actions)
        q2 = self._critic_2(states, actions)

        q1_loss = nn.functional.mse_loss(q1, q_target)
        q2_loss = nn.functional.mse_loss(q2, q_target)
        loss = q1_loss + q2_loss
        return loss

    def _update_critic(self, states, actions, rewards, dones, next_states):
        loss = self._critic_loss(states, actions, rewards, dones, next_states)
        self._critic_1_optimizer.zero_grad()
        self._critic_2_optimizer.zero_grad()
        loss.backward()
        self._critic_1_optimizer.step()
        self._critic_2_optimizer.step()
        return loss.item()

    def _update_actor(self, states, actions):
        loss = self._actor_loss(states, actions)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        return loss.item()

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        critic_loss = self._update_critic(states, actions, rewards, dones, next_states)
        actor_loss = self._update_actor(states, actions)

        soft_update(self._target_critic_1, self._critic_1, self._tau)
        soft_update(self._target_critic_2, self._critic_2, self._tau)

        result = {"critic_loss": critic_loss, "actor_loss": actor_loss}
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self._actor.state_dict(),
            "critic_1": self._critic_1.state_dict(),
            "critic_2": self._critic_2.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._actor.load_state_dict(state_dict["actor"])
        self._critic_1.load_state_dict(state_dict["critic_1"])
        self._critic_2.load_state_dict(state_dict["critic_2"])

def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


# Load demonstrations
def load_demonstrations(path, env_id):
    dataset = ManiSkillTrajectoryDataset(dataset_file=f"{path}/{env_id}/teleop/trajectory.state.pd_joint_delta_pos.h5")
    demo_observations = np.array(dataset.data["traj_0"]["obs"]).squeeze(1)[:-1]
    demo_next_observations = np.array(dataset.data["traj_0"]["obs"]).squeeze(1)[1:]
    demo_actions = np.array(dataset.data["traj_0"]["actions"])
    demo_rewards = np.array(dataset.data["traj_0"]["rewards"])
    demo_dones =  np.array(dataset.data["traj_0"]["terminated"])
    for eps_id in range(1,len(dataset.episodes)):
        eps = dataset.episodes[eps_id]
        trajectory = dataset.data[f"traj_{eps['episode_id']}"]
        demo_observations = np.concatenate((demo_observations, np.array(trajectory["obs"]).squeeze(1)[:-1]), axis=0)
        demo_next_observations = np.concatenate((demo_next_observations, np.array(trajectory["obs"]).squeeze(1)[1:]), axis=0)
        demo_actions = np.concatenate((demo_actions, np.array(trajectory["actions"])), axis=0)
        demo_rewards = np.concatenate((demo_rewards, np.array(trajectory["rewards"])), axis=0)
        demo_dones = np.concatenate((demo_dones, np.array(trajectory["terminated"])), axis=0)
    
    print(f"Demo data loaded with {len(demo_observations)} samples")
    demo_observations = demo_observations[demo_dones==0]
    demo_next_observations = demo_next_observations[demo_dones==0]
    demo_actions = demo_actions[demo_dones==0]
    demo_rewards = demo_rewards[demo_dones==0].reshape(-1,1)
    demo_dones = demo_dones[demo_dones==0].reshape(-1,1)
    print(f"Demo data loaded with {len(demo_observations)} samples")
    print(demo_observations.shape,demo_actions.shape,demo_next_observations.shape,demo_dones.shape,demo_rewards.shape)

    return (demo_observations, demo_actions, demo_rewards, demo_next_observations, demo_dones)


def save_checkpoint(actor, critic_1, critic_2, step, checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic1_state_dict': critic_1.state_dict(),
        'critic2_state_dict': critic_2.state_dict(),
        'step': step
    }, os.path.join(checkpoint_path, f'checkpoint_{step}.pt'))

def load_model(actor, critic1, critic2, path):
    checkpoint = torch.load(path)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic1.load_state_dict(checkpoint['critic1_state_dict'])
    critic2.load_state_dict(checkpoint['critic2_state_dict'])
 
    return actor


def train(args: Args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(obs_mode="state", control_mode="pd_joint_delta_pos", render_mode="rgb_array", sim_backend="gpu")

    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs) 

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate and args.checkpoint!="":
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
   
   
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, **env_kwargs)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.partial_reset, **env_kwargs)
    assert isinstance(envs.unwrapped.single_action_space, gym.spaces.Box), "only continuous action space is supported"


    # Initialize Actor and Critic
    actor = Actor(envs).to(device)
    critic_1 = Critic(envs).to(device)
    critic_2 = Critic(envs).to(device)

    # Replay buffer initialization and load demonstrations
    replay_buffer = ReplayBuffer(
        args,
        args.buffer_size,
        envs.unwrapped.single_observation_space,
        envs.unwrapped.single_action_space,
        device,
        n_envs=args.num_envs
    )


    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    critic_1_optimizer = optim.Adam(critic_1.parameters(), lr=args.q_lr)
    critic_2_optimizer = optim.Adam(critic_2.parameters(), lr=args.q_lr)

    awac = AdvantageWeightedActorCritic(actor, actor_optimizer, critic_1, critic_1_optimizer, critic_2, critic_2_optimizer, gamma=args.gamma, tau=args.tau)


    if not args.evaluate:            
        # Load demonstrations
        obs, actions, rewards, next_obs, dones = load_demonstrations(args.demos_path, args.env_id)        
        print(f"Loaded {len(obs)} demonstrations")
        replay_buffer.add(obs, actions, rewards, next_obs, dones)


    if args.checkpoint!="":
        actor= load_model(actor, critic_1, critic_2, args.checkpoint)
        print(f"Model loaded from {args.checkpoint}")
    
    # Additional tracking for episodic rewards and lengths
    episodic_return = torch.zeros(args.num_envs).to(device)
    episodic_length = torch.zeros(args.num_envs).to(device)
    done_count = 1
    avg_return = 0
    avg_length = 0
    success = 0
    eval_rewards = []

    start_time = time.time()
    if args.evaluate:
        obs, _ = eval_envs.reset(seed=args.seed)
    else:
        obs, _ = envs.reset(seed=args.seed)

    # Training loop
    for global_step in range(args.total_timesteps):
        step = global_step * args.num_envs
        if not args.evaluate:

            actions = actor.act(obs)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            dones = truncations | terminations
            episodic_return += rewards
            episodic_length += 1

            if dones.any():
                avg_return += torch.sum(episodic_return[dones]).item()
                avg_length += torch.sum(episodic_length[dones]).item()
                success += torch.sum(infos["success"]).item()                
                done_count += torch.sum(dones).item()
                episodic_return[dones] = 0
                episodic_length[dones] = 0
            
            # replay_buffer.add(obs.cpu().detach().numpy(), actions.cpu().detach().numpy(), rewards.cpu().detach().numpy(), next_obs.cpu().detach().numpy(), dones.cpu().detach().numpy())
                
            obs = next_obs.clone()

            # Perform update every episode end
            batch = replay_buffer.sample(args.batch_size)
            result = awac.update(batch)
            critic_loss = result["critic_loss"] 
            actor_loss = result["actor_loss"]

            if global_step % 200 == 0 and writer is not None:
                writer.add_scalar("Loss/Critic", critic_loss, step)
                writer.add_scalar("Loss/Actor", actor_loss, step)
                writer.add_scalar("Performance/Average Return", avg_return/done_count, step)
                writer.add_scalar("Performance/Average Length", avg_length/done_count, step)
                writer.add_scalar("Performance/Success Rate", success/done_count, step)
                writer.add_scalar("charts/SPS", int(step / (time.time() - start_time)), step)
                done_count = avg_return = avg_length = success = 0

            if step % args.model_save_interval == 0:
                save_checkpoint(actor, critic_1, critic_2, step, f"runs/{run_name}")

        else:
            actions = actor.act(obs)

            next_obs, rewards, terminations, truncations, infos = eval_envs.step(actions)
            dones = terminations|truncations

            if args.num_eval_envs == 1:
                eval_rewards.append(rewards.item())
            else:
                eval_rewards.append(rewards.mean().item())
            
            if dones.any():
                print(f"env_step={step}, episodic_return={sum(eval_rewards)}")
                eval_rewards = []
                next_obs, _ = eval_envs.reset(seed=args.seed)
                print("success = ", infos["success"])
            
            obs = next_obs.clone()

    if not args.evaluate:
        envs.close()
        wandb.finish()
        writer.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.buffer_size = int(args.buffer_size)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if not args.evaluate:
        if args.track:
            import wandb
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                group=args.exp_name,
                save_code=True,
            )
            print(f"Logging to wandb in project {args.wandb_project_name} with name {run_name}")
        writer = SummaryWriter(f"runs/{run_name}")
        print(f"Logging to tensorboard in runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = None
        eval_rewards = []
        print("Evaluation mode is activated, will not track the experiment")

    train(args)
