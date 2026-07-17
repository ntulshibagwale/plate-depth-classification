import numpy as np
import torch
import random

def set_seeds(random_seed):
    """ Maximum reproducability, same weight init for models """
    torch.backends.cudnn.deterministic = True
    #os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    
    return
    
