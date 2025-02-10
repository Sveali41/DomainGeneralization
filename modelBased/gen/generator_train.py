import torch
from torch import nn
import os
import sys
sys.path.append('/home/siyao/project/rlPractice/DomainGeneralization')
from path import Paths
from modelBased.gen.data.datamodule import GenDataModule
# from basic_gen import Generator, Discriminator
import hydra
from modelBased.common.utils import PROJECT_ROOT, get_env
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from gen import GAN
from modelBased.common.utils import GENERATOR_PATH
import wandb

@hydra.main(version_base=None, config_path=str(GENERATOR_PATH / "conf"), config_name="config")
def train(cfg: DictConfig):
    hparams = cfg
    use_wandb = cfg.training_generator.use_wandb
    # data
    dataloader = GenDataModule(hparams = hparams.training_generator)
    # model
    if hparams.training_generator.generator == "deconv":
        from deconv_gen import Generator, Discriminator
        model = GAN(generator=Generator(hparams.training_generator.z_shape, hparams.training_generator.map_width, hparams.training_generator.map_height, len(hparams.training_generator.map_element)), 
                    discriminator=Discriminator(len(hparams.training_generator.map_element), hparams.training_generator.map_width,  hparams.training_generator.map_height), 
                    z_size=hparams.training_generator.z_shape, lr=hparams.training_generator.lr, wd=hparams.training_generator.wd)
        
    elif cfg.training_generator.model == "basic":
        from basic_gen import Generator, Discriminator
        model = GAN(generator=Generator(hparams.basic.z_shape, hparams.basic.dropout), discriminator=Discriminator(hparams.basic.input_channels, hparams.basic.dropout), z_size=hparams.basic.z_shape, lr=0.0002, wd=0.0)

    # Set up logger
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(project="Gen Training", log_model=True)  
        wandb_logger.experiment.watch(model, log='all', log_freq=1000)
    else:
        print("Debug mode enabled. WandB logging is disabled.")

    # Define the trainer
    metric_to_monitor = 'd_loss' #"loss"
    early_stop_callback = EarlyStopping(monitor=metric_to_monitor, min_delta=0.01, patience=10, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
                            save_top_k=1,
                            monitor = metric_to_monitor,
                            mode = "min",
                            dirpath = get_env('GENERATOR_MODEL_PATH'),
                            filename ="gen-{epoch:02d}-{g_loss:.4f}",
                            verbose = True
                        )
    trainer = pl.Trainer(logger=wandb_logger if use_wandb else None,
                    max_epochs=hparams.training_generator.n_epochs, 
                    gpus=1,
                    callbacks=[early_stop_callback, checkpoint_callback])     
    # Start the training
    trainer.fit(model,dataloader)
    # Log the trained model
    model_pth = hparams.training_generator.pth_path
    trainer.save_checkpoint(model_pth)
    if use_wandb:
        wandb.save(str(model_pth))

@hydra.main(version_base=None, config_path=str(GENERATOR_PATH / "conf"), config_name="config")
def validate(cfg: DictConfig):
    hparams = cfg
    from common.utils import load_gen
    model = load_gen(cfg)
    batch_size = hparams.training_generator.batch_size
    num_tests = 20
    with torch.no_grad():
        # torch.manual_seed(32) 
        for i in range(num_tests):
            z = torch.randn(batch_size,hparams.training_generator.z_shape)
            generated_maps = model(z)
            generated_maps = torch.argmax(generated_maps, dim=1)
            print(generated_maps)
    pass

if __name__ == "__main__":
    # train()
    validate()
    pass
