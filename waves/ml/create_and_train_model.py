import wandb

from waves.ml.training import holdout, holdout_with_valid_and_test
from waves.ml.training import autoencoder_training
from waves.ml.util import set_seeds
from waves.ml.training import holdout_classic_ML, holdout_CNN

def execute(config=None,project=None,tag=None): 
    if tag==None:
        tag = 'bayesian'
    """ main function to launch ml training """
    with wandb.init(project=project, config=config, tags=[tag]): # Begin log        
        config = wandb.config
        set_seeds(config.random_seed)
        device = config.device
        print('---------- HYPERPARAMETERS -----------------------------------')
        for key,value in config.items():
            print(key, ':', value)
        print("")
        print(f'Device: {device}\n')
        print('---------- TRAINING ------------------------------------------')
        # Train and validate model using chosen method
        if config.validation_method == 'holdout':
            holdout(config, device)
        elif config.validation_method == 'holdout_CNN':
            holdout_CNN(config, device)
        elif config.validation_method == 'holdout_with_valid_and_test':
            holdout_with_valid_and_test(config, device)
        elif config.validation_method == 'autoencoder_training':
            autoencoder_training(config, device)
        
    
    return 


def execute_classic_ML(config=None,project=None,tag=None): 
    with wandb.init(project=project, config=config, tags=[tag]): # Begin log        
        config = wandb.config
        device = config.device
        print('---------- HYPERPARAMETERS -----------------------------------')
        for key,value in config.items():
            print(key, ':', value)
        print("")
        print(f'Device: {device}\n')
        print('---------- TRAINING ------------------------------------------')
        # Train and validate model using chosen method
        if config.validation_method == 'holdout':
            holdout_classic_ML(config, device)

    
    return 

