import os
import wandb
from pickle import dump
import torch

from waves.ml.acoustic_emission_dataset import AcousticEmissionDataset, AutoEncoderDataset


def get_train_data(config):
    """ Load training dataset as pytorch dataset. Saves preprocessing. """
    print('Getting training dataset...')
    if config.validation_method == 'autoencoder_training':
        train = AutoEncoderDataset(config.train_path)
                                         
    else:    
        train = AcousticEmissionDataset(config.train_path,
                                         config.sig_len,
                                         config.dt,
                                         config.low_pass,
                                         config.high_pass,
                                         config.feature)
    
    train_scaler = train.scaler # if any scaling performed on train data
    
    # Save preprocessing steps if needed (ex. StandardScaler())
    if train_scaler is not None:
        print('Saving preprocessing scaler...\n')
        dump(train_scaler, open(os.path.join(wandb.run.dir, "scaler.pkl"),
                            'wb'))
        wandb.save('scaler.pkl') 
        
    example_x, example_y, _ = train[0] # to determine feature dim
    feature_dim = example_x.shape[0]   # for model creation input dim
    output_dim = example_y.shape[0]   # for model creation output dim
    
    print("")
    print("Training set loaded in...")
    print(f'train_scaler: {train_scaler}')
    print(f'feature_dim : {feature_dim}')
    print(f'output_dim : {output_dim}\n')

    return train, train_scaler, feature_dim, output_dim
    

def get_valid_data(config, train_scaler):
    """ Load validation dataset as pytorch dataset """
    print('Getting validation dataset...')    
    valid = AcousticEmissionDataset(config.valid_path,
                                     config.sig_len,
                                     config.dt,
                                     config.low_pass,
                                     config.high_pass,
                                     config.feature,
                                     scaler = train_scaler)
    
    return valid


def get_test_data(config, train_scaler=None): 
    """ Load test dataset as pytorch dataset """
    print('Getting test dataset...')
    test = AcousticEmissionDataset(config.test_path,
                                         config.sig_len,
                                         config.dt,
                                         config.low_pass,
                                         config.high_pass,
                                         config.feature,
                                         scaler = train_scaler)

    example_x, example_y, index = test[0] # to determine feature dim

    feature_dim = example_x.shape[0]      # for model creation input dim
    output_dim = example_y.shape[0]   # for model creation output dim
    
    print("")
    print("Test set loaded in...")
    print(f'feature_dim : {feature_dim}')
    print(f'output_dim : {output_dim}\n')

    return test, feature_dim, output_dim


def convert_dataset_to_dataloader(dataset, batch_size): 
    """ Converts dataset object to a data loader (shuffled batching) """
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True,
                                         num_workers=2)
    
    return loader