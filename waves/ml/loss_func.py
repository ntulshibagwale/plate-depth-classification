from torch import nn

def set_loss_func(config):
    """ Set loss function for quantifying model prediction error """
    print("Making the loss function...\n")
    if config.loss_func == 'MSE':
        print('Using MSE\n')
        loss_func = nn.MSELoss()
    elif config.loss_func == 'CrossEntropy':
        print('Using CrossEntropy\n')
        loss_func = nn.CrossEntropyLoss()
        
    return loss_func
