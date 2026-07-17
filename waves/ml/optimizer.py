import torch

def set_optimizer(config, model):
    """ Sets optimizer with model parameters and learning rate """
    print("Making optimizer...\n")
    if config.optimizer_alg == 'Adam':
        print('Using Adam\n')
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate)
        
    return optimizer
