import os
import random
import time
from typing import Any, Dict, List, Optional
from config import Args
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from buffer import Demos_ReplayBuffer as ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import wandb

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset

TensorBatch = List[torch.Tensor]

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Define actor and critic architectures (fill with appropriate layers and dimensions)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cuda") -> np.ndarray:
        state = torch.tensor(state, device=device, dtype=torch.float32)
        return self(state)

class BC:
    def __init__(
        self,
        max_action: np.ndarray,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, _, _, _ = batch

        # Compute actor loss
        pi = self.actor(state)
        actor_loss = F.mse_loss(pi, action)
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]


def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)
    
def save_checkpoint(actor,step, checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'step': step
    }, os.path.join(checkpoint_path, f'checkpoint_{step}.pt'))

def load_model(actor,  path):
    checkpoint = torch.load(path)
    actor.load_state_dict(checkpoint['actor_state_dict'])
 
    return actor

# Load demonstrations
def load_demonstrations(path, env_id):
    dataset = ManiSkillTrajectoryDataset(dataset_file=f"{path}/{env_id}/teleop/trajectory.state.pd_joint_delta_pos.h5")
    demo_data = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': [], 'dones': []}
    for eps_id in range(len(dataset.episodes)):
        trajectory = dataset.data[f"traj_{eps_id}"]
        demo_data['observations'].append(trajectory["obs"][:-1].squeeze(1))
        demo_data['actions'].append(trajectory["actions"])
        demo_data['rewards'].append(trajectory["rewards"])
        demo_data['next_observations'].append(trajectory["obs"][1:].squeeze(1))
        demo_data['dones'].append(trajectory["terminated"])
    for key in demo_data:
        demo_data[key] = np.concatenate(demo_data[key], axis=0)
        if key != 'dones':
            demo_data[key] = torch.tensor(demo_data[key], dtype=torch.float32)
        else:
            # Ensure dones are flat
            demo_data[key] = torch.tensor(demo_data[key], dtype=torch.float32).reshape(-1)
    return (demo_data['observations'], demo_data['actions'], demo_data['rewards'], demo_data['next_observations'], demo_data['dones'])

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


    max_action = torch.as_tensor(envs.unwrapped.single_action_space.high, dtype=torch.float32, device="cuda")
    state_dim = np.array(envs.unwrapped.single_observation_space.shape).prod()
    action_dim =  np.prod(envs.unwrapped.single_action_space.shape)

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    actor = Actor(state_dim, action_dim, max_action).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": args.gamma,
        "device": device,
    }

    print("---------------------------------------")
    print(f"Training BC, Env: {args.env_id}, Seed: {args.seed}")
    print("---------------------------------------")

    # Initialize policy
    trainer = BC(**kwargs)

    # Replay buffer initialization and load demonstrations
    replay_buffer = ReplayBuffer(
        args,
        args.buffer_size,
        envs.unwrapped.single_observation_space,
        envs.unwrapped.single_action_space,
        device,
        n_envs=args.num_envs
    )

    if not args.evaluate:            
        # Load demonstrations
        obs, actions, rewards, next_obs, dones = load_demonstrations(args.demos_path, args.env_id)        
        print(f"Loaded {len(obs)} demonstrations")
        replay_buffer.add(obs, next_obs, actions, rewards, dones)


    if args.checkpoint!="":
        actor= load_model(actor, args.checkpoint)
        print(f"Model loaded from {args.checkpoint}")
    
    # Additional tracking for episodic rewards and lengths
    episodic_return = torch.zeros(args.num_envs).to(device)
    episodic_length = torch.zeros(args.num_envs).to(device)
    done_count = 1
    avg_return = 0
    avg_length = 0
    success = 0
    eval_rewards = []

    if args.evaluate:
        obs, _ = eval_envs.reset(seed=args.seed)
    else:
        obs, _ = envs.reset(seed=args.seed)

    # Training loop
    for step in range(args.total_timesteps):

        if not args.evaluate:
            
            # Perform update every episode end
            batch = replay_buffer.sample(args.batch_size)
            output = trainer.train(batch)
            # critic_loss = output["critic_loss"]
            actor_loss = output["actor_loss"]

            actions = actor.act(obs)

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            dones = terminations|truncations
            

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
                # rewards[dones] += 100

            # replay_buffer.add(obs.cpu().detach().numpy(), next_obs.cpu().detach().numpy(), actions.cpu().detach().numpy(), rewards.cpu().detach().numpy(), terminations.cpu().detach().numpy())

            obs = next_obs.clone()
           
            # Logging
            wandb.log({
                # "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "episodic_reward": episodic_return,
                "episodic_length": episodic_length
            })
            # writer.add_scalar("Loss/Critic", critic_loss, step)
            writer.add_scalar("Loss/Actor", actor_loss, step)
            writer.add_scalar("Performance/Average Return", avg_return/done_count, step)
            writer.add_scalar("Performance/Average Length", avg_length/done_count, step)
            writer.add_scalar("Performance/Success Rate", success/done_count, step)

            if step % args.model_save_interval == 0:
                save_checkpoint(actor, step, f"runs/{run_name}")


        else:

            actions = actor.act(obs)

            next_obs, rewards, terminations, truncations, infos = eval_envs.step(actions.cpu().detach().numpy())
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
