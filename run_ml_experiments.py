import os
import sys
import wandb

# Boiler plate code to import waves module
waves_dir = r'waves'
print(f"Waves module stored at: {waves_dir}")
if sys.path[0] != os.path.dirname(waves_dir):
    sys.path.insert(0, os.path.dirname(waves_dir))
print(f"Directory added to path: {os.path.dirname(waves_dir)}")
print("Waves modules can be imported.")

from waves.ml.create_and_train_model import execute, execute_classic_ML
       

def overfit(wandb_project,dataset):
    """ Overfit entire train set """
    train_path = dataset+'/overfit/10/10_train.json'
    valid_path = ''
    test_path = dataset+'/overfit/10/10_train.json'
    features = ['alpha', 'beta', 'gamma', 'delta', 'epsilon',
                'zeta', 'eta', 'theta', 'iota']
    random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
    for seed in random_seeds:
        for feature in features:
            run_config = {
              "feature" : feature,
              "sig_len" : 2048,
              "model_architecture" : 'NeuralNetwork_Linear_ReLU',
              "hidden_layers" : 5,  # 1,3,5
              "hidden_units" : 50,  # 20, 50
              "epochs" : 100,        # 10, 50, 100
              "random_seed" : seed,
              "device" : "cpu",
              "loss_func" : 'CrossEntropy',#'MSE',
              "optimizer_alg": 'Adam',
              "learning_rate" : 0.001, # 0.01, 0.001
              "batch_size" : 25,
              "dt" : 10**-7,
              "low_pass" : 0,
              "high_pass" : 120*10**4,
              "train_path" : train_path,
              "valid_path" : valid_path,
              "test_path" : test_path,
              "validation_method" : 'holdout', 
              "kfold_random_seed" : None,
              "kfold_n_splits": None,
              "pretrained_model" : None, 
              "task" : 'classification',
              "classes" : ['top','sid','bot']
              }
            execute(config=run_config, project=wandb_project,
                    tag=dataset+"overfit")

    return 
        
def overfit_multiple(wandb_project,dataset):
    """ Overfit entire train set """
    
    ## DISTANCE --------------------------------------------------------------
    train_path = dataset+'/overfit/10/10_train.json'
    valid_path = ''
    test_path = dataset+'/overfit/10/10_train.json'
    features = ['alpha', 'beta', 'gamma', 'delta', 'epsilon',
                'zeta', 'eta', 'theta', 'iota']
    wandb_project = 'DISTANCE_OVERFIT_M5'
    random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
    for seed in random_seeds:
        for feature in features:
            run_config = {
              "feature" : feature,
              "sig_len" : 2048,
              "model_architecture" : 'NeuralNetwork_Linear_ReLU',
              "hidden_layers" : 5,  # 1,3,5
              "hidden_units" : 50,  # 20, 50
              "epochs" : 100,        # 10, 50, 100
              "random_seed" : seed,
              "device" : "cpu",
              "loss_func" : 'CrossEntropy',#'MSE',
              "optimizer_alg": 'Adam',
              "learning_rate" : 0.001, # 0.01, 0.001
              "batch_size" : 25,
              "dt" : 10**-7,
              "low_pass" : 0,
              "high_pass" : 120*10**4,
              "train_path" : train_path,
              "valid_path" : valid_path,
              "test_path" : test_path,
              "validation_method" : 'holdout', 
              "kfold_random_seed" : None,
              "kfold_n_splits": None,
              "pretrained_model" : None, 
              "task" : 'classification',
              "classes" : ['top','sid','bot']
              }
            execute(config=run_config, project=wandb_project,
                    tag=dataset+"overfit")
            
    wandb_project = 'DISTANCE_OVERFIT_M1'
    random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
    for seed in random_seeds:
        for feature in features:
            run_config = {
              "feature" : feature,
              "sig_len" : 2048,
              "model_architecture" : 'NeuralNetwork_Linear_ReLU',
              "hidden_layers" : 1,  # 1,3,5
              "hidden_units" : 20,  # 20, 50
              "epochs" : 10,        # 10, 50, 100
              "random_seed" : seed,
              "device" : "cpu",
              "loss_func" : 'CrossEntropy',#'MSE',
              "optimizer_alg": 'Adam',
              "learning_rate" : 0.01, # 0.01, 0.001
              "batch_size" : 25,
              "dt" : 10**-7,
              "low_pass" : 0,
              "high_pass" : 120*10**4,
              "train_path" : train_path,
              "valid_path" : valid_path,
              "test_path" : test_path,
              "validation_method" : 'holdout', 
              "kfold_random_seed" : None,
              "kfold_n_splits": None,
              "pretrained_model" : None, 
              "task" : 'classification',
              "classes" : ['top','sid','bot']
              }
            execute(config=run_config, project=wandb_project,
                    tag=dataset+"overfit")
            
    wandb_project = 'DISTANCE_OVERFIT_M2'
    random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
    for seed in random_seeds:
        for feature in features:
            run_config = {
              "feature" : feature,
              "sig_len" : 2048,
              "model_architecture" : 'NeuralNetwork_Linear_ReLU',
              "hidden_layers" : 1,  # 1,3,5
              "hidden_units" : 20,  # 20, 50
              "epochs" : 50,        # 10, 50, 100
              "random_seed" : seed,
              "device" : "cpu",
              "loss_func" : 'CrossEntropy',#'MSE',
              "optimizer_alg": 'Adam',
              "learning_rate" : 0.01, # 0.01, 0.001
              "batch_size" : 25,
              "dt" : 10**-7,
              "low_pass" : 0,
              "high_pass" : 120*10**4,
              "train_path" : train_path,
              "valid_path" : valid_path,
              "test_path" : test_path,
              "validation_method" : 'holdout', 
              "kfold_random_seed" : None,
              "kfold_n_splits": None,
              "pretrained_model" : None, 
              "task" : 'classification',
              "classes" : ['top','sid','bot']
              }
            execute(config=run_config, project=wandb_project,
                    tag=dataset+"overfit")

    wandb_project = 'DISTANCE_OVERFIT_M3'
    random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
    for seed in random_seeds:
        for feature in features:
            run_config = {
              "feature" : feature,
              "sig_len" : 2048,
              "model_architecture" : 'NeuralNetwork_Linear_ReLU',
              "hidden_layers" : 3,  # 1,3,5
              "hidden_units" : 20,  # 20, 50
              "epochs" : 10,        # 10, 50, 100
              "random_seed" : seed,
              "device" : "cpu",
              "loss_func" : 'CrossEntropy',#'MSE',
              "optimizer_alg": 'Adam',
              "learning_rate" : 0.01, # 0.01, 0.001
              "batch_size" : 25,
              "dt" : 10**-7,
              "low_pass" : 0,
              "high_pass" : 120*10**4,
              "train_path" : train_path,
              "valid_path" : valid_path,
              "test_path" : test_path,
              "validation_method" : 'holdout', 
              "kfold_random_seed" : None,
              "kfold_n_splits": None,
              "pretrained_model" : None, 
              "task" : 'classification',
              "classes" : ['top','sid','bot']
              }
            execute(config=run_config, project=wandb_project,
                    tag=dataset+"overfit")
    
    wandb_project = 'DISTANCE_OVERFIT_M4'
    random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
    for seed in random_seeds:
        for feature in features:
            run_config = {
              "feature" : feature,
              "sig_len" : 2048,
              "model_architecture" : 'NeuralNetwork_Linear_ReLU',
              "hidden_layers" : 3,  # 1,3,5
              "hidden_units" : 20,  # 20, 50
              "epochs" : 50,        # 10, 50, 100
              "random_seed" : seed,
              "device" : "cpu",
              "loss_func" : 'CrossEntropy',#'MSE',
              "optimizer_alg": 'Adam',
              "learning_rate" : 0.01, # 0.01, 0.001
              "batch_size" : 25,
              "dt" : 10**-7,
              "low_pass" : 0,
              "high_pass" : 120*10**4,
              "train_path" : train_path,
              "valid_path" : valid_path,
              "test_path" : test_path,
              "validation_method" : 'holdout', 
              "kfold_random_seed" : None,
              "kfold_n_splits": None,
              "pretrained_model" : None, 
              "task" : 'classification',
              "classes" : ['top','sid','bot']
              }
            execute(config=run_config, project=wandb_project,
                    tag=dataset+"overfit")
   

    train_path = 'coupling/overfit/10/10_train.json'
    valid_path = ''
    test_path = 'coupling/overfit/10/10_train.json'
    features = ['alpha', 'beta', 'gamma', 'delta', 'epsilon',
                'zeta', 'eta', 'theta', 'iota']
    wandb_project = 'COUPLING_OVERFIT_M2'
    random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
    for seed in random_seeds:
        for feature in features:
            run_config = {
              "feature" : feature,
              "sig_len" : 2048,
              "model_architecture" : 'NeuralNetwork_Linear_ReLU',
              "hidden_layers" : 1,  # 1,3,5
              "hidden_units" : 20,  # 20, 50
              "epochs" : 10,        # 10, 50, 100
              "random_seed" : seed,
              "device" : "cpu",
              "loss_func" : 'CrossEntropy',#'MSE',
              "optimizer_alg": 'Adam',
              "learning_rate" : 0.01, # 0.01, 0.001
              "batch_size" : 25,
              "dt" : 10**-7,
              "low_pass" : 0,
              "high_pass" : 120*10**4,
              "train_path" : train_path,
              "valid_path" : valid_path,
              "test_path" : test_path,
              "validation_method" : 'holdout', 
              "kfold_random_seed" : None,
              "kfold_n_splits": None,
              "pretrained_model" : None, 
              "task" : 'classification',
              "classes" : ['top','sid','bot']
              }
            execute(config=run_config, project=wandb_project,
                    tag=dataset+"overfit")
    wandb_project = 'COUPLING_OVERFIT_M3'
    random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
    for seed in random_seeds:
        for feature in features:
            run_config = {
              "feature" : feature,
              "sig_len" : 2048,
              "model_architecture" : 'NeuralNetwork_Linear_ReLU',
              "hidden_layers" : 3,  # 1,3,5
              "hidden_units" : 20,  # 20, 50
              "epochs" : 10,        # 10, 50, 100
              "random_seed" : seed,
              "device" : "cpu",
              "loss_func" : 'CrossEntropy',#'MSE',
              "optimizer_alg": 'Adam',
              "learning_rate" : 0.01, # 0.01, 0.001
              "batch_size" : 25,
              "dt" : 10**-7,
              "low_pass" : 0,
              "high_pass" : 120*10**4,
              "train_path" : train_path,
              "valid_path" : valid_path,
              "test_path" : test_path,
              "validation_method" : 'holdout', 
              "kfold_random_seed" : None,
              "kfold_n_splits": None,
              "pretrained_model" : None, 
              "task" : 'classification',
              "classes" : ['top','sid','bot']
              }
            execute(config=run_config, project=wandb_project,
                    tag=dataset+"overfit")
    wandb_project = 'COUPLING_OVERFIT_M4'
    random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
    for seed in random_seeds:
        for feature in features:
            run_config = {
              "feature" : feature,
              "sig_len" : 2048,
              "model_architecture" : 'NeuralNetwork_Linear_ReLU',
              "hidden_layers" : 3,  # 1,3,5
              "hidden_units" : 20,  # 20, 50
              "epochs" : 50,        # 10, 50, 100
              "random_seed" : seed,
              "device" : "cpu",
              "loss_func" : 'CrossEntropy',#'MSE',
              "optimizer_alg": 'Adam',
              "learning_rate" : 0.01, # 0.01, 0.001
              "batch_size" : 25,
              "dt" : 10**-7,
              "low_pass" : 0,
              "high_pass" : 120*10**4,
              "train_path" : train_path,
              "valid_path" : valid_path,
              "test_path" : test_path,
              "validation_method" : 'holdout', 
              "kfold_random_seed" : None,
              "kfold_n_splits": None,
              "pretrained_model" : None, 
              "task" : 'classification',
              "classes" : ['top','sid','bot']
              }
            execute(config=run_config, project=wandb_project,
                    tag=dataset+"overfit")
            
    return 
        


def bayesian(wandb_project,dataset):
    """ Bayesian optimization of neural network hyperparameters using valid"""
    train_path = dataset+'/bayesian/bayesian_train.json'
    valid_path = dataset+'/bayesian/bayesian_valid.json'
    test_path =  dataset+'/bayesian/bayesian_test.json'
    features = ['alpha', 'beta', 'gamma', 'delta', 'epsilon',
                'zeta', 'eta', 'theta', 'iota']
    #features = ['theta','iota']
    #features=['eta' ]

    for feature in features:
        sweep_config = {
          "name" : f"{feature}",
          "method" : "bayes",
          'metric' : {"goal":"minimize",
                      "name":'holdout_results.holdout_valid_loss'},
          "parameters" : {
              "feature" : {"value" : feature},   
              "sig_len" : {"value" : 2048},
              "model_architecture" : {"value" : 'NeuralNetwork_Linear_ReLU'},
              "hidden_layers" : {"values" : [1,3,5]}, 
              "hidden_units" : {"values" : [20,50,100,150]},
              "epochs" : {"values" : [10,50,100,300]},
              "random_seed" : {"value" : int(42*22)},
              "device" : {"value" : 'cpu'},
              "loss_func" : {"value" : 'CrossEntropy'},
              "optimizer_alg": {"value" : 'Adam'},
              "learning_rate" :{'max': 0.01, 'min': 0.0001},
              "batch_size" : {"values" : [10,25,100]},
              "dt" : {"value" : 10**-7},
              "low_pass" : {"value" : 0},
              "high_pass" : {"value" : 120*10**4},
              "train_path" : {"value" : train_path},
              "valid_path" : {"value" : valid_path},
              "test_path" : {"value" : test_path},
              "validation_method" : {"value" : 'holdout_with_valid_and_test'},
              "kfold_random_seed" : {"value" : None},
              "kfold_n_splits" : {"value" : None},
              "pretrained_model" : {"value":None},    
              "task" : {"value" : 'classification'},
              "classes" : {"value" : ['top','sid','bot']}
              }
          }
    
        # Create sweep (only run once)
        sweep_id = wandb.sweep(sweep_config,project=wandb_project) 
        #sweep_id = 'dw20ndpd' # Enter sweep id in case it stops mid run
        wandb.agent(sweep_id, entity="ae-ml_ucsb",
                    project = wandb_project,
                    function=execute, count=100)
        
    return

        
def stratified(wandb_project,dataset):
    """ Run stratified where each coupling is evaluated as a test set """
    train_path = [dataset+'/stratified/01/01_train.json',
                  dataset+'/stratified/02/02_train.json',
                  dataset+'/stratified/03/03_train.json',
                  dataset+'/stratified/04/04_train.json',
                  dataset+'/stratified/05/05_train.json',
                  dataset+'/stratified/06/06_train.json',
                  dataset+'/stratified/07/07_train.json',
                  dataset+'/stratified/08/08_train.json',
                  dataset+'/stratified/09/09_train.json',
                  dataset+'/stratified/10/10_train.json']     
    valid_path = ''
    test_path = [dataset+'/stratified/01/01_test.json',
                 dataset+'/stratified/02/02_test.json',
                 dataset+'/stratified/03/03_test.json',
                 dataset+'/stratified/04/04_test.json',
                 dataset+'/stratified/05/05_test.json',
                 dataset+'/stratified/06/06_test.json',
                 dataset+'/stratified/07/07_test.json',
                 dataset+'/stratified/08/08_test.json',
                 dataset+'/stratified/09/09_test.json',
                 dataset+'/stratified/10/10_test.json']  
    features = ['alpha', 'beta', 'gamma', 'delta', 'epsilon',
                'zeta', 'eta', 'theta', 'iota']
    random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
    for seed in random_seeds:
        for feature in features:
            for idx,_ in enumerate(train_path):
                run_config = {
                  "feature" : feature,
                  "sig_len" : 2048,
                  "model_architecture" : 'NeuralNetwork_Linear_ReLU',
                  "hidden_layers" : 5, # 1,3,5
                  "hidden_units" : 50, # 20, 50
                  "epochs" : 100,       # 10,50,100
                  "random_seed" : seed,
                  "device" : "cpu",
                  "loss_func" : 'CrossEntropy',#'MSE',
                  "optimizer_alg": 'Adam',
                  "learning_rate" : 0.001, # 0.01, 0.001
                  "batch_size" : 25,
                  "dt" : 10**-7,
                  "low_pass" : 0,
                  "high_pass" : 120*10**4,
                  "train_path" : train_path[idx],
                  "valid_path" : valid_path,
                  "test_path" : test_path[idx],
                  "validation_method" : 'holdout', 
                  "kfold_random_seed" : None,
                  "kfold_n_splits": None,
                  "pretrained_model" : None, 
                  "task" : 'classification',
                  "classes" : ['top','sid','bot']
                  }
                execute(config=run_config, project=wandb_project, 
                        tag=dataset+"-stratified")
            
    return 


def stratified_top_sid(wandb_project,dataset):
    """ Run stratified where each coupling is evaluated as a test set """
    train_path = [dataset+'/stratified_top_sid/01/01_train.json',
                  dataset+'/stratified_top_sid/02/02_train.json',
                  dataset+'/stratified_top_sid/03/03_train.json',
                  dataset+'/stratified_top_sid/04/04_train.json',
                  dataset+'/stratified_top_sid/05/05_train.json',
                  dataset+'/stratified_top_sid/06/06_train.json',
                  dataset+'/stratified_top_sid/07/07_train.json',
                  dataset+'/stratified_top_sid/08/08_train.json',
                  dataset+'/stratified_top_sid/09/09_train.json',
                  dataset+'/stratified_top_sid/10/10_train.json']     
    valid_path = ''
    test_path = [dataset+'/stratified_top_sid/01/01_test.json',
                 dataset+'/stratified_top_sid/02/02_test.json',
                 dataset+'/stratified_top_sid/03/03_test.json',
                 dataset+'/stratified_top_sid/04/04_test.json',
                 dataset+'/stratified_top_sid/05/05_test.json',
                 dataset+'/stratified_top_sid/06/06_test.json',
                 dataset+'/stratified_top_sid/07/07_test.json',
                 dataset+'/stratified_top_sid/08/08_test.json',
                 dataset+'/stratified_top_sid/09/09_test.json',
                 dataset+'/stratified_top_sid/10/10_test.json']  
    features = ['alpha', 'beta', 'gamma', 'delta', 'epsilon',
                'zeta', 'eta', 'theta', 'iota']
    #features = ['epsilon','zeta','eta','theta','iota']
    random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
#    random_seeds=[int(42*15)]
    for seed in random_seeds:
        for feature in features:
            for idx,_ in enumerate(train_path):
                run_config = {
                  "feature" : feature,
                  "sig_len" : 2048,
                  "model_architecture" : 'NeuralNetwork_Linear_ReLU',
                  "hidden_layers" : 1, # 1,3,5
                  "hidden_units" : 20, # 20, 50
                  "epochs" : 10,       # 10,50,100
                  "random_seed" : seed,
                  "device" : "cpu",
                  "loss_func" : 'CrossEntropy',#'MSE',
                  "optimizer_alg": 'Adam',
                  "learning_rate" : 0.01, # 0.01, 0.001
                  "batch_size" : 25,
                  "dt" : 10**-7,
                  "low_pass" : 0,
                  "high_pass" : 120*10**4,
                  "train_path" : train_path[idx],
                  "valid_path" : valid_path,
                  "test_path" : test_path[idx],
                  "validation_method" : 'holdout', 
                  "kfold_random_seed" : None,
                  "kfold_n_splits": None,
                  "pretrained_model" : None, 
                  "task" : 'classification',
                  "classes" : ['top','sid']#,'bot']
                  }
                execute(config=run_config, project=wandb_project, 
                        tag=dataset+"-stratified")
            
    return


def stratified_top_bot(wandb_project,dataset):
    """ Run stratified where each coupling is evaluated as a test set """
    train_path = [dataset+'/stratified_top_bot/01/01_train.json',
                  dataset+'/stratified_top_bot/02/02_train.json',
                  dataset+'/stratified_top_bot/03/03_train.json',
                  dataset+'/stratified_top_bot/04/04_train.json',
                  dataset+'/stratified_top_bot/05/05_train.json',
                  dataset+'/stratified_top_bot/06/06_train.json',
                  dataset+'/stratified_top_bot/07/07_train.json',
                  dataset+'/stratified_top_bot/08/08_train.json',
                  dataset+'/stratified_top_bot/09/09_train.json',
                  dataset+'/stratified_top_bot/10/10_train.json']     
    valid_path = ''
    test_path = [dataset+'/stratified_top_bot/01/01_test.json',
                 dataset+'/stratified_top_bot/02/02_test.json',
                 dataset+'/stratified_top_bot/03/03_test.json',
                 dataset+'/stratified_top_bot/04/04_test.json',
                 dataset+'/stratified_top_bot/05/05_test.json',
                 dataset+'/stratified_top_bot/06/06_test.json',
                 dataset+'/stratified_top_bot/07/07_test.json',
                 dataset+'/stratified_top_bot/08/08_test.json',
                 dataset+'/stratified_top_bot/09/09_test.json',
                 dataset+'/stratified_top_bot/10/10_test.json']  
    features = ['alpha', 'beta', 'gamma', 'delta', 'epsilon',
                'zeta', 'eta', 'theta', 'iota']
    #features = ['epsilon','zeta','eta','theta','iota']
    random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
#    random_seeds=[int(42*15)]
    for seed in random_seeds:
        for feature in features:
            for idx,_ in enumerate(train_path):
                run_config = {
                  "feature" : feature,
                  "sig_len" : 2048,
                  "model_architecture" : 'NeuralNetwork_Linear_ReLU',
                  "hidden_layers" : 1, # 1,3,5
                  "hidden_units" : 20, # 20, 50
                  "epochs" : 10,       # 10,50,100
                  "random_seed" : seed,
                  "device" : "cpu",
                  "loss_func" : 'CrossEntropy',#'MSE',
                  "optimizer_alg": 'Adam',
                  "learning_rate" : 0.01, # 0.01, 0.001
                  "batch_size" : 25,
                  "dt" : 10**-7,
                  "low_pass" : 0,
                  "high_pass" : 120*10**4,
                  "train_path" : train_path[idx],
                  "valid_path" : valid_path,
                  "test_path" : test_path[idx],
                  "validation_method" : 'holdout', 
                  "kfold_random_seed" : None,
                  "kfold_n_splits": None,
                  "pretrained_model" : None, 
                  "task" : 'classification',
                  "classes" : ['top','sid']#,'bot']
                  }
                execute(config=run_config, project=wandb_project, 
                        tag=dataset+"-stratified")
            
    return
      
def stratified_cnn(wandb_project,dataset):
    """ Run stratified where each coupling is evaluated as a test set """
    train_path = [dataset+'/stratified/01/01_train.json',
                  dataset+'/stratified/02/02_train.json',
                  dataset+'/stratified/03/03_train.json',
                  dataset+'/stratified/04/04_train.json',
                  dataset+'/stratified/05/05_train.json',
                  dataset+'/stratified/06/06_train.json',
                  dataset+'/stratified/07/07_train.json',
                  dataset+'/stratified/08/08_train.json',
                  dataset+'/stratified/09/09_train.json',
                  dataset+'/stratified/10/10_train.json']     
    valid_path = ''
    test_path = [dataset+'/stratified/01/01_test.json',
                 dataset+'/stratified/02/02_test.json',
                 dataset+'/stratified/03/03_test.json',
                 dataset+'/stratified/04/04_test.json',
                 dataset+'/stratified/05/05_test.json',
                 dataset+'/stratified/06/06_test.json',
                 dataset+'/stratified/07/07_test.json',
                 dataset+'/stratified/08/08_test.json',
                 dataset+'/stratified/09/09_test.json',
                 dataset+'/stratified/10/10_test.json']  
    features = ['iota']
    random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
    for seed in random_seeds:
        for feature in features:
            for idx,_ in enumerate(train_path):
                run_config = {
                  "feature" : feature,
                  "sig_len" : 2048,
                  "model_architecture" : 'NeuralNetwork_Conv1D_C1',
                  "hidden_layers" : 5, # 1,3,5
                  "hidden_units" : 50, # 20, 50
                  "epochs" : 20,       # 10,50,100
                  "random_seed" : seed,
                  "device" : "cpu",
                  "loss_func" : 'CrossEntropy',#'MSE',
                  "optimizer_alg": 'Adam',
                  "learning_rate" : 0.001, # 0.01, 0.001
                  "batch_size" : 25,
                  "dt" : 10**-7,
                  "low_pass" : 0,
                  "high_pass" : 120*10**4,
                  "train_path" : train_path[idx],
                  "valid_path" : valid_path,
                  "test_path" : test_path[idx],
                  "validation_method" : 'holdout', 
                  "kfold_random_seed" : None,
                  "kfold_n_splits": None,
                  "pretrained_model" : None, 
                  "task" : 'classification',
                  "classes" : ['top','sid','bot']
                  }
                execute(config=run_config, project=wandb_project, 
                        tag=dataset+"stratified")
            
    return 


def stratified_cnn_specific(wandb_project,dataset):
    """ Run stratified where each coupling is evaluated as a test set """
    train_path = [dataset+'/stratified/01/01_train.json',
                  ]     
    valid_path = ''
    test_path = [dataset+'/stratified/01/01_test.json',
                ]  
    features = ['iota']
    random_seeds=[int(42*1)] 
    for seed in random_seeds:
        for feature in features:
            for idx,_ in enumerate(train_path):
                run_config = {
                  "feature" : feature,
                  "sig_len" : 2048,
                  "model_architecture" : 'NeuralNetwork_Conv1D_C1',
                  "hidden_layers" : 5, # 1,3,5
                  "hidden_units" : 50, # 20, 50
                  "epochs" : 20,       # 10,50,100
                  "random_seed" : seed,
                  "device" : "cpu",
                  "loss_func" : 'CrossEntropy',#'MSE',
                  "optimizer_alg": 'Adam',
                  "learning_rate" : 0.001, # 0.01, 0.001
                  "batch_size" : 25,
                  "dt" : 10**-7,
                  "low_pass" : 0,
                  "high_pass" : 120*10**4,
                  "train_path" : train_path[idx],
                  "valid_path" : valid_path,
                  "test_path" : test_path[idx],
                  "validation_method" : 'holdout_CNN', 
                  "kfold_random_seed" : None,
                  "kfold_n_splits": None,
                  "pretrained_model" : None, 
                  "task" : 'classification',
                  "classes" : ['top','sid','bot']
                  }
                execute(config=run_config, project=wandb_project, 
                        tag=dataset+"stratified")
            
    return 


def entire(wandb_project,dataset):
    """ Entire dataset split into train and test """
    train_path = [dataset+'/entire/2%/2%_train.json',
                  dataset+'/entire/10%/10%_train.json',
                  dataset+'/entire/100%/100%_train.json']
    valid_path = ''
    test_path = dataset+'/entire/test.json'
    features = ['alpha', 'beta', 'gamma', 'delta', 'epsilon',
                'zeta', 'eta', 'theta', 'iota']
    random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
    for seed in random_seeds:
        for feature in features:
            for idx,_ in enumerate(train_path):
                run_config = {
                  "feature" : feature,
                  "sig_len" : 2048,
                  "model_architecture" : 'NeuralNetwork_Linear_ReLU',
                  "hidden_layers" : 5,  # 1,3,5
                  "hidden_units" : 250,  # 20, 50
                  "epochs" : 100,        # 10, 50, 100
                  "random_seed" : seed,
                  "device" : "cpu",
                  "loss_func" : 'CrossEntropy',#'MSE',
                  "optimizer_alg": 'Adam',
                  "learning_rate" : 0.001, # 0.01, 0.001
                  "batch_size" : 25,
                  "dt" : 10**-7,
                  "low_pass" : 0,
                  "high_pass" : 120*10**4,
                  "train_path" : train_path[idx],
                  "valid_path" : valid_path,
                  "test_path" : test_path,
                  "validation_method" : 'holdout', 
                  "kfold_random_seed" : None,
                  "kfold_n_splits": None,
                  "pretrained_model" : None, 
                  "task" : 'classification',
                  "classes" : ['top','sid','bot']
                  }
                execute(config=run_config, project=wandb_project,
                        tag=dataset+"entire")
    
    return 

def entire_cnn(wandb_project,dataset):
    """ Entire dataset split into train and test """
    train_path = [dataset+'/entire/2%/2%_train.json',
                  dataset+'/entire/10%/10%_train.json',
                  dataset+'/entire/100%/100%_train.json']
    valid_path = ''
    test_path = dataset+'/entire/test.json'
    features = ['iota']
    #random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
    random_seeds=[int(42*1)] 

    for seed in random_seeds:
        for feature in features:
            for idx,_ in enumerate(train_path):
                run_config = {
                  "feature" : feature,
                  "sig_len" : 2048,
                  "model_architecture" : 'NeuralNetwork_Conv1D_01',
                  "hidden_layers" : 5,  # 1,3,5
                  "hidden_units" : 250,  # 20, 50
                  "epochs" : 50,        # 10, 50, 100
                  "random_seed" : seed,
                  "device" : "cpu",
                  "loss_func" : 'CrossEntropy',#'MSE',
                  "optimizer_alg": 'Adam',
                  "learning_rate" : 0.001, # 0.01, 0.001
                  "batch_size" : 5,
                  "dt" : 10**-7,
                  "low_pass" : 0,
                  "high_pass" : 120*10**4,
                  "train_path" : train_path[idx],
                  "valid_path" : valid_path,
                  "test_path" : test_path,
                  "validation_method" : 'holdout', 
                  "kfold_random_seed" : None,
                  "kfold_n_splits": None,
                  "pretrained_model" : None, 
                  "task" : 'classification',
                  "classes" : ['top','sid','bot']
                  }
                execute(config=run_config, project=wandb_project,
                        tag=dataset+"entire")
    
    return 


def entire_cnn(wandb_project,dataset):
    """ Entire dataset split into train and test """
    train_path = [dataset+'/entire/2%/2%_train.json',
                  dataset+'/entire/10%/10%_train.json',
                  dataset+'/entire/100%/100%_train.json']
    valid_path = ''
    test_path = dataset+'/entire/test.json'
    features = ['iota']
    random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)] 
    for seed in random_seeds:
        for feature in features:
            for idx,_ in enumerate(train_path):
                run_config = {
                  "feature" : feature,
                  "sig_len" : 2048,
                  "model_architecture" : 'FCN',
                  "hidden_layers" : 5,  # 1,3,5
                  "hidden_units" : 250,  # 20, 50
                  "epochs" : 100,        # 10, 50, 100
                  "random_seed" : seed,
                  "device" : "cpu",
                  "loss_func" : 'CrossEntropy',#'MSE',
                  "optimizer_alg": 'Adam',
                  "learning_rate" : 0.001, # 0.01, 0.001
                  "batch_size" : 25,
                  "dt" : 10**-7,
                  "low_pass" : 0,
                  "high_pass" : 120*10**4,
                  "train_path" : train_path[idx],
                  "valid_path" : valid_path,
                  "test_path" : test_path,
                  "validation_method" : 'holdout', 
                  "kfold_random_seed" : None,
                  "kfold_n_splits": None,
                  "pretrained_model" : None, 
                  "task" : 'classification',
                  "classes" : ['top','sid','bot']
                  }
                execute(config=run_config, project=wandb_project,
                        tag=dataset+"entire")
    
    return 
    

def cam(wandb_project,dataset):
    """ Run when testing extrapolation from coupling dataset to dataset """
    run_config = {
      "feature" : 'iota', #iota
      "sig_len" : 2048,
      "model_architecture" : 'NeuralNetwork_Conv1D',
      "hidden_layers" : 2, 
      "hidden_units" : 20,
      "epochs" : 300,
      "random_seed" : 5,
      "device" : "cpu",
      "loss_func" : 'CrossEntropy',#'MSE',
      "optimizer_alg": 'Adam',
      "learning_rate" : 0.01,
      "batch_size" : 5,
      "dt" : 10**-7,
      "low_pass" : 0,
      "high_pass" : 120*10**4,
      "train_path" : 'cam/cam_train.json',
      "valid_path" : "",
      "test_path" : 'cam/cam_test.json',
      "validation_method" : 'holdout', 
      "kfold_random_seed" : None,
      "kfold_n_splits": None,
      "pretrained_model" : None, 
      "task" : 'classification',
      "classes" : ['top','sid','bot']
      }
    execute(config=run_config, project=wandb_project,tag="cam")
        
    return 


def stratified_classic_ML(wandb_project,dataset):
    train_path = [dataset+'/stratified/01/01_train.json',
                  dataset+'/stratified/02/02_train.json',
                  dataset+'/stratified/03/03_train.json',
                  dataset+'/stratified/04/04_train.json',
                  dataset+'/stratified/05/05_train.json',
                  dataset+'/stratified/06/06_train.json',
                  dataset+'/stratified/07/07_train.json',
                  dataset+'/stratified/08/08_train.json',
                  dataset+'/stratified/09/09_train.json',
                  dataset+'/stratified/10/10_train.json']     
    valid_path = ''
    test_path = [dataset+'/stratified/01/01_test.json',
                 dataset+'/stratified/02/02_test.json',
                 dataset+'/stratified/03/03_test.json',
                 dataset+'/stratified/04/04_test.json',
                 dataset+'/stratified/05/05_test.json',
                 dataset+'/stratified/06/06_test.json',
                 dataset+'/stratified/07/07_test.json',
                 dataset+'/stratified/08/08_test.json',
                 dataset+'/stratified/09/09_test.json',
                 dataset+'/stratified/10/10_test.json']  
    features = ['alpha', 'beta', 'gamma', 'delta', 'epsilon',
                'zeta', 'eta', 'theta', 'iota']
    #random_seeds=[int(42*1),int(42*22),int(42*13),int(42*72),int(42*15)]
    models = ['SVM','Logistic','RF']
    
    for model in models:
        for feature in features:
            for idx,_ in enumerate(train_path):
                run_config = {
                  "feature" : feature,
                  "device": "cpu",
                  "sig_len" : 2048,
                  "model_architecture" : model,
                  "dt" : 10**-7,
                  "low_pass" : 0,
                  "high_pass" : 120*10**4,
                  "train_path" : train_path[idx],
                  "valid_path" : valid_path,
                  "test_path" : test_path[idx],
                  "validation_method" : 'holdout', 
                  "task" : 'classification',
                  "classes" : ['top','sid','bot']
                  }
                execute_classic_ML(config=run_config, project=wandb_project, 
                        tag=dataset+"-stratified")
            
    return 





def pretrain(wandb_project,dataset):
    """ Run when testing extrapolation from coupling dataset to dataset """
    run_config = {
      "feature" : 'iota', #iota
      "sig_len" : 2048,
      "model_architecture" : 'AutoEncoder_Conv1D_AE1',
      "hidden_layers" : 2, 
      "hidden_units" : 20,
      "epochs" : 300,
      "random_seed" : 5,
      "device" : "cpu",
      "loss_func" : 'MSE',#'MSE',
      "optimizer_alg": 'Adam',
      "learning_rate" : 0.001,
      "batch_size" : 100,
      "dt" : 10**-7,
      "low_pass" : 0,
      "high_pass" : 120*10**4,
      "train_path" : dataset+'/morscher.json',
      "valid_path" : "",
      "test_path" : '',
      "validation_method" : 'autoencoder_training', 
      "kfold_random_seed" : None,
      "kfold_n_splits": None,
      "pretrained_model" : None, 
      "task" : 'classification',
      "classes" : ['top','sid','bot']
      }
    execute(config=run_config, project=wandb_project,tag="pretrain")
        
    return 


def main():
    wandb_project= 'stratified_CNN'
    dataset = 'distance'
    #dataset = 'coupling'
    #stratified_top_sid(wandb_project, dataset)
    #stratified_top_bot(wandb_project, dataset)
    stratified_cnn(wandb_project, dataset)
    #stratified_classic_ML(wandb_project,dataset)
    #stratified_cnn_specific(wandb_project, dataset)
    #pretrain(wandb_project, dataset)
    #entire(wandb_project, dataset)
    #overfit_multiple(wandb_project, dataset)
    #entire_cnn(wandb_project, dataset)
    #stratified_cnn(wandb_project, dataset)
    #stratified_distance(wandb_project)
    #extrapolation(wandb_project)
    #bayesian(wandb_project,dataset)
    #cam(wandb_project, dataset)
    
    return
    
    
if __name__=='__main__':
    main()