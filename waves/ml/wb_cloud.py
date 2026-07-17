import wandb
from pickle import load
import os
import shutil
import torch
from torch import nn

from waves.ml.model import create_model

def save_file_to_wandb(file_path):
    """ Save file from file_path to cloud """
    file_name = os.path.basename(os.path.normpath(file_path))
    dst = os.path.join(wandb.run.dir, file_name)
    shutil.copyfile(file_path, dst) # copy plot to work dir
    wandb.save(file_name)
    
    return


def get_run(run_file_path):
    """ Get hyperparameters / configuration dict for specified W&B run """
    api = wandb.Api()
    run = api.run(run_file_path)

    return run.config, run.summary


def load_train_scaler(run_file_path):
    """ Load scaler used on the train data from pickle file """
    scaler_pkl_wb_path = wandb.restore('scaler.pkl', run_path=run_file_path)
    train_scaler = load(open(scaler_pkl_wb_path, 'rb'))
    
    return train_scaler


def load_trained_model(run_file_path, config, feature_dim, output_dim,
                       model_name):
    """ Load trained model from wandb run path """
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = create_model(config, device, feature_dim, output_dim)
    model_pth_wb_path = wandb.restore(model_name, run_path=run_file_path)
    model.load_state_dict(torch.load(model_pth_wb_path.name)) 
    print('LOADED MODEL WEIGHTS :\n')
    for idx,m in enumerate(model.layers):
        if type(m) == nn.Linear:
          print(m)
          print(m.weight)
          print("")

    return model 


def log_training(train_loss, valid_loss, example_ct, epoch):
    """ Log training outputs to wandb """
    wandb.log({"epoch": epoch, "train_loss": train_loss, 
               'valid_loss': valid_loss}, step=example_ct)
    print(f"Epoch: {epoch} | Example Count: "  + str(example_ct).zfill(5) \
          + f"| Train Loss: {train_loss:.5f} | Valid Loss: {valid_loss:.5f}")

    return


    