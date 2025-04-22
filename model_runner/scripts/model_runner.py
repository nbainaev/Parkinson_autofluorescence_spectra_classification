import wandb
import yaml
import os

from trainer import DiscriminatorTrainer

# specify you login in wandb, you should also authorize with token first, see wandb quick start documentation
os.environ['WANDB_ENTITY'] = 'nik-baynaev-national-research-nuclear-university-mephi'

config_path = './model_runner/configs/runner_conf.yaml'


if __name__ == '__main__':
    # load configs

    config = dict()

    with open(config_path, 'r') as file:
        config['run'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['trainer_data_conf'], 'r') as file:
        config['data'] = yaml.load(file, Loader=yaml.Loader)
    
    with open(config['run']['model_conf'], 'r') as file:
        config['model'] = yaml.load(file, Loader=yaml.Loader)
    
    with open(config['run']['train_conf'], 'r') as file:
        config['train'] = yaml.load(file, Loader=yaml.Loader)

    if config['run']['logging']:
        # start wandb logger
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ['WANDB_ENTITY'],
            config=config
        )
    else:
        logger = None

    trainer = DiscriminatorTrainer(logger, config)
    trainer.train()