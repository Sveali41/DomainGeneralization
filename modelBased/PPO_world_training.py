import sys
sys.path.append('/home/siyao/project/rlPractice/DomainGeneralization')
from path import Paths
from minigrid_custom_env import CustomEnvFromFile
from minigrid.wrappers import FullyObsWrapper
import torch
import torch.nn as nn
import numpy as np
import torch.distributions as td
from PPO import PPO
import os
import hydra
from modelBased.common.utils import PROJECT_ROOT
from datetime import datetime
from modelBased.world_model import SimpleNN
from modelBased.world_model_training import normalize, map_obs_to_nearest_value
from omegaconf import DictConfig, OmegaConf

# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

def get_destination(obs, episode, maxstep, destination):
    """
    from the obs state, check if the agent has reached the destination
    and return done and reward

    1.object:("unseen": 0,  "empty": 1, "wall": 2, "door": 4, "key": 5, "goal": 8, "agent": 10)
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10

    2. color:
    "red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5

    3. status
    State, 0: open, 1: closed, 2: locked

    check from wrappers.py full_obs-->encode
    """
    if obs[destination[0], destination[1]][0] == 10:
        # agent has reached the destination
        if episode >= maxstep:
            done = True
            reward = 0
        else:
            reward = 1 - 0.9 * (episode / maxstep)
            done = True
    else:
        done = False
        reward = 0
    return done, reward


def find_position(array, target):
    """
    Find the position of a target value in a 3D numpy array.
    
    Args:
        array (np.ndarray): The 3D array to search.
        target (tuple): The target value to locate (e.g., (8, 1, 0)).

    Returns:
        tuple: The position (x, y) of the target in the array if found, otherwise None.
    """
    # Find all indices where the value matches the target
    result = np.argwhere((array == target).all(axis=-1))

    # Check if any matches were found
    if result.size > 0:
        return tuple(result[0])  # Return the first match as a tuple (x, y)
    else:
        return None
    
@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "conf/model"), config_name="config")
def training_agent(cfg: DictConfig):
    hparams = cfg
    
    # 1. world Model
    model = SimpleNN(hparams=hparams).to(device)
    checkpoint = torch.load(hparams.world_model.pth_folder)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # 2. PPO
    # hyperparameters
    start_time = datetime.now().replace(microsecond=0)
    lr_actor = hparams.PPO.lr_actor
    lr_critic = hparams.PPO.lr_critic
    gamma = hparams.PPO.gamma
    K_epochs = hparams.PPO.K_epochs
    eps_clip = hparams.PPO.eps_clip
    action_std = hparams.PPO.action_std
    action_std_decay_rate = hparams.PPO.action_std_decay_rate
    min_action_std = hparams.PPO.min_action_std
    action_std_decay_freq = hparams.PPO.action_std_decay_freq
    max_training_timesteps = hparams.PPO.max_training_timesteps
    save_model_freq = hparams.PPO.save_model_freq
    max_ep_len = hparams.PPO.max_ep_len
    has_continuous_action_space = hparams.PPO.has_continuous_action_space
    checkpoint_path = hparams.PPO.checkpoint_path
    
    # 3. Real env
    path = Paths()
    env = FullyObsWrapper(
        CustomEnvFromFile(txt_file_path=path.LEVEL_FILE, custom_mission="Find the key and open the door.",
                          max_steps=2000,
                          render_mode="rgb"))
    
    # 4. Initialize training
    i_episode = 0
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    print_freq = max_ep_len * 4
    print_running_reward = 0
    print_running_episodes = 0
    time_step = 0
    
    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space
    else:
        action_dim = env.action_space.n
    state_dim = np.prod(env.observation_space['image'].shape)
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)
    
    # training loop
    while time_step <= max_training_timesteps:
        state = env.reset()[0]['image']
        goal_position = find_position(state (8, 1, 0)) # find the goal position
        current_ep_reward = 0
        for t in range(1, max_ep_len + 1):
            # self.buffer.states = [state.squeeze(0) if state.dim() > 1 else state for state in self.buffer.states]
            if t==1:
                state = normalize(state).to(device)
            else:
                state = state.squeeze()
            action = ppo_agent.select_action(state)
            state = model(state, torch.tensor(action/hparams.world_model.action_norm_values).to(device).unsqueeze(0))
            state = state.to(dtype=torch.float32)
            # denorm the state
            state_denorm = map_obs_to_nearest_value(cfg, state)
            # obtain reward from the state representation & done
            done, reward = get_destination(state_denorm, t, max_ep_len, goal_position)
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                if use_wandb:
                    wandb.log({"average_reward": print_avg_reward})

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1

    env.close()


if __name__ == "__main__":
    use_wandb = True
    if use_wandb:
        import wandb

        wandb.login(key="ae0b0db53ae05bebce869b5ccc77b9efd0d62c73")
        wandb.init(project='WM PPO', entity='svea41')

    training_agent()

    if use_wandb:
        wandb.finish()
