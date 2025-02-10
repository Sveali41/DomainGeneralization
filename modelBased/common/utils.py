import os
from pathlib import Path
from typing import Dict, List, Optional
from typing import Sequence
import dotenv
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch

def get_env(env_name: str, default: Optional[str] = None) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.
    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable
    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            raise KeyError(f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value

def load_envs(env_file: Optional[str] = '.env') -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.
    It is possible to define all the system specific variables in the `env_file`.
    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


load_envs()

def denormalize(x, width=3, height=3):
    """Denormalize the obs dimension data from its flattened state.
        input: x: torch.tensor of shape (,54)
    """
    obs_norm_values = [10, 5, 3] # Example normalization values for 3 channels
    # Reshape the data to its original shape before flattening
    x = x.reshape(width, height ,3)
    
    # Ensure that the norm_values is not None and has the correct length
    if obs_norm_values is None or len(obs_norm_values) != x.shape[-1]:
        raise ValueError("Normalization values must be provided and must match the number of channels in the data.")
    
    # Denormalize each channel using the provided norm values
    for i in range(x.shape[-1]):  # Loop over the last dimension (channels)
        max_value = obs_norm_values[i]
        if max_value != 0:  # Avoid multiplication by zero (though normally this wouldn't be an issue)
            x[:, :, i] = x[:, :, i] * max_value
    return x

def normalize(x):
    """Normalize the obs data and flatten it.
        input: x: np.array of shape (6,3,3)
    """
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32) 
    obs_norm_values = [10, 5, 3] 
    # Ensure that the norm_values is not None and has the correct length
    if obs_norm_values is None or len(obs_norm_values) != x.shape[-1]:
        raise ValueError("Normalization values must be provided and must match the number of channels in the data.")
    # Normalize each channel using the provided norm values
    for i in range(x.shape[-1]):  # Loop over the last dimension (channels)
        max_value = obs_norm_values[i]
        if max_value != 0:  # Avoid division by zero
            x[:, :, i] = x[:, :, i] / max_value  
    # Flatten the data
    x = x.reshape(-1)
    x = torch.tensor(x)
    return x


def map_to_nearest_value(tensor, valid_values):
    valid_values = torch.tensor(valid_values, dtype=torch.float32).to(tensor.device)
    tensor = tensor.unsqueeze(-1)  # Add a dimension to compare with valid values
    differences = torch.abs(tensor - valid_values)  # Calculate differences
    indices = torch.argmin(differences, dim=-1)  # Get index of nearest value
    nearest_values = valid_values[indices]  # Get nearest values using indices
    return nearest_values

def map_obs_to_nearest_value(cfg, obs):
    """
    Maps each value in the tensor to the nearest valid value from the valid_values list.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (,54).
        valid_values (list): From config read list of valid values to map to.
        
    Returns:
        torch.Tensor: 1.Denorm the tensor 2.Each element is replaced 
                      by the nearest valid value.
    """
    # Denormalize the tensor
    obs_denorm = denormalize(obs.clone())
    # load the valid values from the config
    hparams = cfg
    valid_values_obj = hparams.world_model.valid_values_obj
    valid_values_color = hparams.world_model.valid_values_color
    valid_values_state = hparams.world_model.valid_values_state
    # Map each channel to the nearest valid value
    obs_denorm[:, :, 0] = map_to_nearest_value(obs_denorm[:, :, 0], valid_values_obj)
    obs_denorm[:, :, 1] = map_to_nearest_value(obs_denorm[:, :, 1], valid_values_color)
    obs_denorm[:, :, 2] = map_to_nearest_value(obs_denorm[:, :, 2], valid_values_state)
    return obs_denorm

PROJECT_ROOT : Path = Path(get_env("PROJECT_ROOT"))
GENERATOR_PATH : Path = Path(get_env("GENERATOR_PATH"))
MAIN_PATH : Path = Path(get_env("MAIN_PATH"))
# TRAINER_PATH : Path = Path(get_env("TRAINER_PATH"))
