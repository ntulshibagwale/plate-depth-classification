import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch import nn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from waves.ml.data import get_train_data, get_test_data, get_valid_data
from waves.ml.loss_func import set_loss_func
from waves.ml.optimizer import set_optimizer
from waves.ml.model import create_model
from waves.visuals.plot_train_valid_loss_curve import \
    plot_train_valid_test_loss_curve, plot_train_test_loss_curve
from waves.misc import flatten
from waves.ml.wb_cloud import load_trained_model
from waves.signal_processing import calc_fft

def holdout_with_valid_and_test(config, device):
    """ Training of model while optimizing on valid, and evaluating on test """
    print("HOLDOUT **********************************************************")
    # Get training dataset, scaler, and data shapes for model creation
    train, train_scaler, feature_dim, output_dim = get_train_data(config)  
    valid = get_valid_data(config, train_scaler)
    test,_,_ = get_test_data(config, train_scaler)

    print("Making data loaders...\n")
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=True)
    # shuffling redundant since done in .json creation


    # Create neural network
    if config.pretrained_model is None:
        model = create_model(config, device, feature_dim, output_dim)
    else:
        model = load_trained_model(config.pretrained_model, config,
                                   feature_dim, output_dim,
                                   'model_holdout.pth')
    
    # Make the loss and optimizer
    loss_func = set_loss_func(config)
    optimizer = set_optimizer(config, model)
    
    # Train 
    train_loss_history, valid_loss_history, test_loss_history, epochs = train_model_with_test(
                                                   config, device,
                                                   model, train_loader, 
                                                   valid_loader, 
                                                   test_loader,
                                                   loss_func, 
                                                   optimizer)
    
    # Save trained model
    torch.save(model.state_dict(),
               os.path.join(wandb.run.dir,'model_holdout.pth')) 
    wandb.save('model_holdout.pth')
    

    train_loss, train_predicted, train_targets, train_indices,\
        train_class_accuracy, train_total_accuracy, train_metrics = \
                get_model_predictions_classification(config, device, model, 
                                         train_loader, loss_func)
    valid_loss, valid_predicted, valid_targets, valid_indices,\
        valid_class_accuracy, valid_total_accuracy, valid_metrics = \
                get_model_predictions_classification(config, device, model, 
                                         valid_loader, loss_func)            
    test_loss, test_predicted, test_targets, test_indices,\
        test_class_accuracy, test_total_accuracy, test_metrics = \
                get_model_predictions_classification(config, device, model, 
                                         test_loader, loss_func) 
    # Log results          
    print('Logging holdout results..\n') 
    holdout_results = {'holdout_train_loss_history': train_loss_history,
               'holdout_train_loss': train_loss,
               'holdout_train_predicted': train_predicted,
               'holdout_train_targets': train_targets,
               'holdout_train_indices': train_indices,
               'holdout_train_class_accuracy': train_class_accuracy,
               'holdout_train_total_accuracy': train_total_accuracy,
               'holdout_train_metrics': train_metrics,
               'holdout_valid_loss_history': valid_loss_history,
               'holdout_valid_loss': valid_loss,
               'holdout_valid_predicted': valid_predicted,
               'holdout_valid_targets': valid_targets,
               'holdout_valid_indices': valid_indices,
               'holdout_valid_class_accuracy': valid_class_accuracy,
               'holdout_valid_total_accuracy': valid_total_accuracy,
               'holdout_valid_metrics': valid_metrics,
               'holdout_test_loss_history': valid_loss_history,
               'holdout_test_loss': test_loss,
               'holdout_test_predicted': test_predicted,
               'holdout_test_targets': test_targets,
               'holdout_test_indices': test_indices,
               'holdout_test_class_accuracy': test_class_accuracy,
               'holdout_test_total_accuracy': test_total_accuracy,
               'holdout_test_metrics': test_metrics}
    wandb.log({'holdout_results': holdout_results})
        
    
    plot_train_valid_test_loss_curve(train_loss_history, valid_loss_history, test_loss_history,
                            epochs, name='Holdout',
                            save_folder = wandb.run.dir)
    wandb.save('loss_curve_holdout.png') 
    
    return


def autoencoder_training(config, device):
    """ Training an autoencoder on unlabeled data """
    train, train_scaler, feature_dim, output_dim = get_train_data(config)  
    
    print("Making data loaders...\n")
    train_loader = DataLoader(train, batch_size=config.batch_size, 
                              shuffle=True) 
    
    # Create autoencoder
    if config.pretrained_model is None:
        model = create_model(config, device, feature_dim, output_dim)
    else:
        model = load_trained_model(config.pretrained_model, config,
                                   feature_dim, output_dim,
                                   'pretrained_model.pth')
        
    # Make the loss and optimizer
    loss_func = set_loss_func(config)
    optimizer = set_optimizer(config, model)
    
    # Train 
    train_loss_history, epochs = pretrain_model(config, device,
                                                   model, train_loader, 
                                                   loss_func, optimizer)
    
    # Save trained model
    torch.save(model.state_dict(),
               os.path.join(wandb.run.dir,'pretrained_model.pth')) 
    wandb.save('pretrained_model.pth')
    train_loss, train_predicted, train_targets, train_indices = \
        get_model_predictions_autoencoder(config, device, model, train_loader,
                                          loss_func)
    # Log results          
    print('Logging autoencoder training results..\n') 
    holdout_results = {'train_loss_history': train_loss_history,
               'train_loss': train_loss,
               'train_predicted': train_predicted,
               'train_targets': train_targets,
               'train_indices': train_indices}
    wandb.log({'results': holdout_results})
        
    plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.plot(train_targets[0])
    ax2.plot(train_predicted[0])
    plt.show()
    
    plot_train_test_loss_curve(train_loss_history, train_loss_history,
                            epochs, name='Autoencoder',
                            save_folder = wandb.run.dir)
    wandb.save('loss_curve.png') 
    
    return
    

def holdout_classic_ML(config, device):
    print("HOLDOUT **********************************************************")

    train, train_scaler, feature_dim, output_dim = get_train_data(config)
    test, _, _ = get_test_data(config, train_scaler)

    X_train = train.x.detach().cpu().numpy()
    y_train = train.y.detach().cpu().numpy().reshape(-1)

    X_test = test.x.detach().cpu().numpy()
    y_test = test.y.detach().cpu().numpy().reshape(-1)

    if config.model_architecture == "SVM":
        model = LinearSVC(
            max_iter=10000,
            random_state=0,
        )

    elif config.model_architecture == "Logistic":
        model = LogisticRegression(
            max_iter=10000,
            random_state=0,
        )

    elif config.model_architecture == "RF":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=0,
        )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_total_accuracy = accuracy_score(y_train, y_train_pred)
    test_total_accuracy = accuracy_score(y_test, y_test_pred)

    train_class_accuracy = {}
    test_class_accuracy = {}

    for class_idx in np.unique(y_train):
        class_idx = int(class_idx)

        train_mask = y_train == class_idx
        test_mask = y_test == class_idx

        train_class_accuracy[class_idx] = accuracy_score(
            y_train[train_mask],
            y_train_pred[train_mask],
        )

        if np.sum(test_mask) > 0:
            test_class_accuracy[class_idx] = accuracy_score(
                y_test[test_mask],
                y_test_pred[test_mask],
            )
        else:
            test_class_accuracy[class_idx] = None

    train_metrics = classification_report(
        y_train,
        y_train_pred,
        output_dict=True,
        zero_division=0,
    )

    test_metrics = classification_report(
        y_test,
        y_test_pred,
        output_dict=True,
        zero_division=0,
    )

    print("Logging holdout results..\n")

    holdout_results = {
        "holdout_train_predicted": y_train_pred.tolist(),
        "holdout_train_targets": y_train.tolist(),
        "holdout_train_class_accuracy": train_class_accuracy,
        "holdout_train_total_accuracy": train_total_accuracy,
        "holdout_train_metrics": train_metrics,

        "holdout_test_predicted": y_test_pred.tolist(),
        "holdout_test_targets": y_test.tolist(),
        "holdout_test_class_accuracy": test_class_accuracy,
        "holdout_test_total_accuracy": test_total_accuracy,
        "holdout_test_metrics": test_metrics,
    }

    wandb.log({
        "holdout_results": holdout_results,
        "holdout_train_accuracy": train_total_accuracy,
        "holdout_test_accuracy": test_total_accuracy,
    })

    return holdout_results


def get_layer_activations(model, x):
    """
    Forward pass that returns activations from each convolution block.
    Assumes the model has c1, c2, c3, c4.
    """

    if x.dim() == 2:
        x = x[:, None, :]

    a1 = model.c1(x)
    a2 = model.c2(a1)
    a3 = model.c3(a2)
    a4 = model.c4(a3)

    activations = {
        "c1": a1,
        "c2": a2,
        "c3": a3,
        "c4": a4,
    }

    return activations


def plot_average_activation_maps(model, test_loader, config, device, save_folder):
    """
    Plot average activation maps for each class and each convolution layer.
    Handles both integer labels and one-hot labels.
    """

    model.eval()

    activation_store = {
        "c1": [],
        "c2": [],
        "c3": [],
        "c4": [],
    }

    target_store = []

    with torch.no_grad():
        for batch in test_loader:

            if len(batch) == 2:
                x, y = batch

            elif len(batch) >= 3:
                x = batch[0]
                y = batch[1]

            x = x.to(device)
            y = y.to(device)

            # ---------------------------------------------------------
            # Convert one-hot labels to integer class labels
            # ---------------------------------------------------------

            if y.dim() > 1:
                y = torch.argmax(y, dim=1)

            activations = get_layer_activations(model, x)

            for layer_name in activation_store.keys():
                activation_store[layer_name].append(
                    activations[layer_name].detach().cpu()
                )

            target_store.append(y.detach().cpu())

    targets = torch.cat(target_store, dim=0).numpy()

    for layer_name in activation_store.keys():

        acts = torch.cat(
            activation_store[layer_name],
            dim=0,
        ).numpy()

        n_classes = len(config.classes)

        fig, axes = plt.subplots(
            n_classes,
            1,
            figsize=(9, 2.4 * n_classes),
            sharex=True,
            sharey=True,
        )

        if n_classes == 1:
            axes = [axes]

        im = None

        for class_idx, class_name in enumerate(config.classes):
            ax = axes[class_idx]

            mask = targets == class_idx

            if np.sum(mask) == 0:
                ax.set_ylabel(class_name)
                continue

            avg_activation = np.mean(
                acts[mask],
                axis=0,
            )

            im = ax.imshow(
                avg_activation,
                aspect="auto",
                origin="lower",
            )

            ax.set_ylabel(class_name)

        axes[-1].set_xlabel("Activation time index")

        fig.suptitle(f"Average activation map: {layer_name}")

        if im is not None:
            fig.colorbar(
                im,
                ax=axes,
                label="Activation",
            )

        out_path = os.path.join(
            save_folder,
            f"average_activation_map_{layer_name}.png",
        )

        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        wandb.save(out_path)
        
        
def plot_class_activation_maps(model, test_loader, config, device, save_folder):
    """
    Class activation maps for NeuralNetwork_Conv1D_C1.

    Uses the final conv layer c4 and the final linear layer weights.
    CAM_c(t) = sum_k w_c,k * A_k(t)
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import wandb

    model.eval()

    class_names = list(config.classes)

    cam_store = {class_name: [] for class_name in class_names}

    with torch.no_grad():
        for batch in test_loader:

            if len(batch) == 2:
                x, y = batch
            else:
                x = batch[0]
                y = batch[1]

            x = x.to(device)
            y = y.to(device)

            if y.dim() > 1:
                y = torch.argmax(y, dim=1)

            if x.dim() == 2:
                x = x[:, None, :]

            # ---------------------------------------------------------
            # Forward manually up to final conv layer
            # ---------------------------------------------------------

            a1 = model.c1(x)
            a2 = model.c2(a1)
            a3 = model.c3(a2)
            a4 = model.c4(a3)

            # a4 shape: batch x channels x time
            # linear weights shape: classes x channels
            weights = model.linear.weight.detach()

            for i in range(x.shape[0]):
                true_class = int(y[i].item())
                class_name = class_names[true_class]

                cam = torch.sum(
                    weights[true_class, :, None] * a4[i],
                    dim=0,
                )

                cam = cam.detach().cpu().numpy()

                cam = cam - np.min(cam)

                if np.max(cam) > 0:
                    cam = cam / np.max(cam)

                cam_store[class_name].append(cam)

    # -------------------------------------------------------------------------
    # Plot average CAM for each class
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        len(class_names),
        1,
        figsize=(9, 2.5 * len(class_names)),
        sharex=True,
        sharey=True,
    )

    if len(class_names) == 1:
        axes = [axes]

    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx]

        cams = cam_store[class_name]

        if len(cams) == 0:
            ax.set_ylabel(class_name)
            continue

        cams = np.asarray(cams)
        mean_cam = np.mean(cams, axis=0)

        time_index = np.arange(len(mean_cam))

        ax.plot(
            time_index,
            mean_cam,
            linewidth=1.5,
        )

        ax.set_ylabel(class_name)

    axes[-1].set_xlabel("CAM time index")

    fig.suptitle("Average class activation maps")

    fig.tight_layout()

    out_path = os.path.join(
        save_folder,
        "average_class_activation_maps.png",
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    wandb.save(out_path)
        
        
def holdout_CNN(config, device):
    """ Training of model while evaluating on separate validation set """
    print("HOLDOUT **********************************************************")
    # Get training dataset, scaler, and data shapes for model creation
    train, train_scaler, feature_dim, output_dim = get_train_data(config)  
    test,_,_ = get_test_data(config, train_scaler)

    print("Making data loaders...\n")
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True) 
    test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=True)
    
    # Create neural network
    if config.pretrained_model is None:
        model = create_model(config, device, feature_dim, output_dim)
    else:
        model = load_trained_model(config.pretrained_model, config,
                                   feature_dim, output_dim,
                                   'model_holdout.pth')
    
    # Make the loss and optimizer
    loss_func = set_loss_func(config)
    optimizer = set_optimizer(config, model)
    
    # Train 
    train_loss_history, test_loss_history, epochs = train_model(
                                                   config, device,
                                                   model, train_loader, 
                                                   test_loader, 
                                                   loss_func, 
                                                   optimizer)
    
    # Save trained model
    torch.save(model.state_dict(),
               os.path.join(wandb.run.dir,'model_holdout.pth')) 
    wandb.save('model_holdout.pth')
    
    train_loss, train_predicted, train_targets, train_indices,\
        train_class_accuracy, train_total_accuracy, train_metrics = \
                get_model_predictions_classification(config, device, model, 
                                         train_loader, loss_func)
    test_loss, test_predicted, test_targets, test_indices,\
        test_class_accuracy, test_total_accuracy, test_metrics = \
                get_model_predictions_classification(config, device, model, 
                                         test_loader, loss_func)            
                
    # Log results          
    print('Logging holdout results..\n') 
    holdout_results = {'holdout_train_loss_history': train_loss_history,
               'holdout_train_loss': train_loss,
               'holdout_train_predicted': train_predicted,
               'holdout_train_targets': train_targets,
               'holdout_train_indices': train_indices,
               'holdout_train_class_accuracy': train_class_accuracy,
               'holdout_train_total_accuracy': train_total_accuracy,
               'holdout_train_metrics': train_metrics,
               'holdout_test_loss_history': test_loss_history,
               'holdout_test_loss': test_loss,
               'holdout_test_predicted': test_predicted,
               'holdout_test_targets': test_targets,
               'holdout_test_indices': test_indices,
               'holdout_test_class_accuracy': test_class_accuracy,
               'holdout_test_total_accuracy': test_total_accuracy,
               'holdout_test_metrics': test_metrics}
    wandb.log({'holdout_results': holdout_results})
        
  
    plot_train_test_loss_curve(
        train_loss_history,
        test_loss_history,
        epochs,
        name='Holdout',
        save_folder=wandb.run.dir
    )

    wandb.save('loss_curve_holdout.png') 

    # ---------------------------------------------------------
    # Interpretability plots
    # ---------------------------------------------------------

    if config.model_architecture == 'NeuralNetwork_Conv1D_C1':

        plot_average_waveform_with_saliency(
            model,
            test_loader,
            config,
            device,
            save_folder=wandb.run.dir,save_individual=False
        )

    return

def plot_average_waveform_with_saliency(
    model,
    test_loader,
    config,
    device,
    save_folder,
    save_individual=False,
):

    model.eval()

    class_names = ["Top", "Side", "Bottom"]

    n_classes = len(class_names)

    class_colors = ["C0", "C1", "C2"]

    waves_by_class = [[] for _ in range(n_classes)]
    saliency_by_class = [[] for _ in range(n_classes)]

    time = np.arange(config.sig_len) * config.dt * 1e6

    individual_dir = os.path.join(
        save_folder,
        "individual_saliency_maps",
    )

    if save_individual:
        os.makedirs(individual_dir, exist_ok=True)

    example_counter = 0

    # -------------------------------------------------------------------------
    # Calculate saliency maps
    # -------------------------------------------------------------------------

    for batch in test_loader:

        x = batch[0]
        y = batch[1]

        x = x.to(device)
        y = y.to(device)

        # Convert one-hot labels to class indices
        if y.dim() > 1:
            y = torch.argmax(y, dim=1)

        # Tell PyTorch to track gradients w.r.t. input waveform
        x = x.clone().detach()
        x.requires_grad = True

        # Forward pass
        output = model(x)

        # Select score for the true class
        true_class_scores = output[
            torch.arange(len(y)),
            y,
        ]

        # Backpropagate to waveform
        score = true_class_scores.sum()

        model.zero_grad()
        score.backward()

        # Extract gradients
        saliency = x.grad.detach().cpu().numpy()

        waves = x.detach().cpu().numpy()
        labels = y.detach().cpu().numpy()

        # ---------------------------------------------------------------------
        # Save individual examples
        # ---------------------------------------------------------------------

        if save_individual:

            for i in range(len(waves)):

                wave = waves[i]

                sal = np.abs(saliency[i])

                label = int(labels[i])

                class_name = class_names[label]

                color = class_colors[label]

                # Normalize saliency
                if np.max(sal) > 0:
                    sal = sal / np.max(sal)

                y_min = np.min(wave)
                y_max = np.max(wave)

                y_pad = 0.08 * (y_max - y_min)

                fig, ax = plt.subplots(figsize=(9, 3))

                # Saliency background
                ax.imshow(
                    sal[None, :],
                    aspect="auto",
                    extent=[
                        time[0],
                        time[-1],
                        y_min - y_pad,
                        y_max + y_pad,
                    ],
                    cmap="Reds",
                    alpha=0.35,
                )

                # Waveform
                ax.plot(
                    time,
                    wave,
                    color=color,
                    linewidth=1.5,
                )

                # Class label
                ax.text(
                    0.02,
                    0.92,
                    class_name,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=12,
                    fontweight="normal",
                )

                ax.set_xlabel("Time (µs)")
                ax.set_ylabel("Amplitude (V)")

                ax.set_ylim(
                    y_min - y_pad,
                    y_max + y_pad,
                )

                out_path = os.path.join(
                    individual_dir,
                    f"saliency_example_{example_counter:04d}_{class_name}.png",
                )

                fig.tight_layout()

                fig.savefig(
                    out_path,
                    dpi=200,
                )

                plt.close(fig)

                example_counter += 1

        # ---------------------------------------------------------------------
        # Store waveforms and saliency maps by class
        # ---------------------------------------------------------------------

        for class_idx in range(n_classes):

            class_mask = labels == class_idx

            if np.sum(class_mask) == 0:
                continue

            waves_by_class[class_idx].append(
                waves[class_mask]
            )

            saliency_by_class[class_idx].append(
                np.abs(saliency[class_mask])
            )

    # -------------------------------------------------------------------------
    # Compute class averages
    # -------------------------------------------------------------------------

    avg_waves = []
    avg_saliencies = []

    for class_idx in range(n_classes):

        class_waves = np.concatenate(
            waves_by_class[class_idx],
            axis=0,
        )

        class_saliencies = np.concatenate(
            saliency_by_class[class_idx],
            axis=0,
        )

        avg_wave = np.mean(
            class_waves,
            axis=0,
        )

        avg_saliency = np.mean(
            class_saliencies,
            axis=0,
        )

        # Normalize saliency
        if np.max(avg_saliency) > 0:
            avg_saliency = avg_saliency / np.max(avg_saliency)

        avg_waves.append(avg_wave)
        avg_saliencies.append(avg_saliency)

    # -------------------------------------------------------------------------
    # Plot average waveform saliency maps
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        n_classes,
        1,
        figsize=(9, 2.8 * n_classes),
        sharex=True,
    )

    if n_classes == 1:
        axes = [axes]

    for class_idx in range(n_classes):

        ax = axes[class_idx]

        avg_wave = avg_waves[class_idx]

        avg_saliency = avg_saliencies[class_idx]

        class_name = class_names[class_idx]

        color = class_colors[class_idx]

        y_min = np.min(avg_wave)
        y_max = np.max(avg_wave)

        y_pad = 0.08 * (y_max - y_min)

        # Saliency background
        ax.imshow(
            avg_saliency[None, :],
            aspect="auto",
            extent=[
                time[0],
                time[-1],
                y_min - y_pad,
                y_max + y_pad,
            ],
            cmap="Reds",
            alpha=0.35,
        )

        # Waveform
        ax.plot(
            time,
            avg_wave,
            color=color,
            linewidth=2.0,
        )

        # Class label
        ax.text(
            0.02,
            0.92,
            class_name,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="normal",
        )

        ax.set_ylabel("Amplitude (V)")

        ax.set_ylim(
            y_min - y_pad,
            y_max + y_pad,
        )

    axes[-1].set_xlabel("Time (µs)")

    fig.tight_layout()

    out_path = os.path.join(
        save_folder,
        "average_waveform_with_saliency.png",
    )

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    wandb.save(out_path)

    # -------------------------------------------------------------------------
    # Upload individual plots to W&B
    # -------------------------------------------------------------------------

    if save_individual:

        for filename in os.listdir(individual_dir):

            if filename.endswith(".png"):

                full_path = os.path.join(
                    individual_dir,
                    filename,
                )

                wandb.log({
                    filename: wandb.Image(full_path)
                })
                
# def plot_average_waveform_with_saliency(model, test_loader, config, device, save_folder):

#     model.eval()

#     class_names = list(config.classes)
#     n_classes = len(class_names)

#     waves_by_class = [[] for _ in range(n_classes)]
#     saliency_by_class = [[] for _ in range(n_classes)]

#     # -------------------------------------------------------------------------
#     # Calculate saliency for each batch
#     # -------------------------------------------------------------------------

#     for batch in test_loader:

#         x = batch[0]
#         y = batch[1]

#         x = x.to(device)
#         y = y.to(device)

#         # If y is one-hot encoded, convert it back to class numbers
#         if y.dim() > 1:
#             y = torch.argmax(y, dim=1)

#         # Tell PyTorch to track gradients with respect to the input waveform
#         x = x.clone().detach()
#         x.requires_grad = True

#         # Forward pass
#         output = model(x)

#         # Select the score for the true class of each waveform
#         true_class_scores = output[torch.arange(len(y)), y]

#         # Backpropagate from the true class scores to the input waveform
#         score = true_class_scores.sum()

#         model.zero_grad()
#         score.backward()

#         # Saliency is the gradient of the class score with respect to the input
#         saliency = x.grad.detach().cpu().numpy()
#         waves = x.detach().cpu().numpy()
#         labels = y.detach().cpu().numpy()

#         # Store waveforms and saliency maps by class
#         for class_idx in range(n_classes):

#             class_mask = labels == class_idx

#             waves_by_class[class_idx].append(waves[class_mask])
#             saliency_by_class[class_idx].append(np.abs(saliency[class_mask]))

#     # -------------------------------------------------------------------------
#     # Average waveforms and saliency maps by class
#     # -------------------------------------------------------------------------

#     avg_waves = []
#     avg_saliencies = []

#     for class_idx in range(n_classes):

#         class_waves = np.concatenate(waves_by_class[class_idx], axis=0)
#         class_saliencies = np.concatenate(saliency_by_class[class_idx], axis=0)

#         avg_wave = np.mean(class_waves, axis=0)
#         avg_saliency = np.mean(class_saliencies, axis=0)

#         # Normalize saliency so the strongest value is 1
#         avg_saliency = avg_saliency / np.max(avg_saliency)

#         avg_waves.append(avg_wave)
#         avg_saliencies.append(avg_saliency)

#     # -------------------------------------------------------------------------
#     # Plot
#     # -------------------------------------------------------------------------

#     time = np.arange(config.sig_len) * config.dt * 1e6

#     fig, axes = plt.subplots(
#         n_classes,
#         1,
#         figsize=(9, 2.8 * n_classes),
#         sharex=True,
#     )

#     if n_classes == 1:
#         axes = [axes]

#     for class_idx in range(n_classes):

#         ax = axes[class_idx]

#         avg_wave = avg_waves[class_idx]
#         avg_saliency = avg_saliencies[class_idx]

#         y_min = np.min(avg_wave)
#         y_max = np.max(avg_wave)
#         y_pad = 0.08 * (y_max - y_min)

#         # Plot saliency as red background
#         ax.imshow(
#             avg_saliency[None, :],
#             aspect="auto",
#             extent=[time[0], time[-1], y_min - y_pad, y_max + y_pad],
#             cmap="Reds",
#             alpha=0.75,
#         )

#         # Plot average waveform
#         ax.plot(
#             time,
#             avg_wave,
#             color="black",
#             linewidth=2.0,
#         )

#         ax.set_ylabel(class_names[class_idx])
#         ax.set_ylim(y_min - y_pad, y_max + y_pad)

#     axes[-1].set_xlabel("Time (µs)")

#     fig.suptitle("Average waveform with saliency")
#     fig.tight_layout()

#     out_path = os.path.join(save_folder, "average_waveform_with_saliency.png")

#     fig.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     wandb.save(out_path)
    
    
def plot_strongest_filters(
    model,
    test_loader,
    config,
    device,
    save_folder,
    layer_name="c1",
    n_filters_to_plot=5,
):
    """
    Find strongest convolution filters according to mean activation energy.

    Then plot:
        1. filter kernels
        2. filter FFTs
        3. mean activation vs time for each class
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import wandb

    model.eval()

    # -------------------------------------------------------------------------
    # Get convolution layer
    # -------------------------------------------------------------------------

    layer = getattr(model, layer_name)

    conv_weights = layer.conv.weight.detach().cpu().numpy()

    # shape:
    # filters x 1 x kernel_size

    kernels = conv_weights[:, 0, :]

    n_filters = kernels.shape[0]

    # -------------------------------------------------------------------------
    # Store activations
    # -------------------------------------------------------------------------

    activation_energy = np.zeros(n_filters)

    activation_store = [[] for _ in range(n_filters)]

    target_store = []

    with torch.no_grad():
        for batch in test_loader:

            if len(batch) == 2:
                x, y = batch
            else:
                x = batch[0]
                y = batch[1]

            x = x.to(device)
            y = y.to(device)

            if y.dim() > 1:
                y = torch.argmax(y, dim=1)

            if x.dim() == 2:
                x = x[:, None, :]

            # -------------------------------------------------------------
            # Forward to requested layer
            # -------------------------------------------------------------

            a1 = model.c1(x)

            if layer_name == "c1":
                acts = a1

            elif layer_name == "c2":
                acts = model.c2(a1)

            elif layer_name == "c3":
                acts = model.c3(model.c2(a1))

            elif layer_name == "c4":
                acts = model.c4(model.c3(model.c2(a1)))

            else:
                raise ValueError("Unknown layer_name")

            acts_np = acts.detach().cpu().numpy()

            # -------------------------------------------------------------
            # Activation energy
            # -------------------------------------------------------------

            # acts shape:
            # batch x filters x time

            activation_energy += np.sum(
                acts_np**2,
                axis=(0, 2),
            )

            for f in range(n_filters):
                activation_store[f].append(
                    acts_np[:, f, :]
                )

            target_store.append(
                y.detach().cpu().numpy()
            )

    targets = np.concatenate(target_store)

    for f in range(n_filters):
        activation_store[f] = np.concatenate(
            activation_store[f],
            axis=0,
        )

    # -------------------------------------------------------------------------
    # Rank strongest filters
    # -------------------------------------------------------------------------

    strongest_indices = np.argsort(
        activation_energy
    )[::-1]

    strongest_indices = strongest_indices[:n_filters_to_plot]

    print("\nStrongest filters")
    print("=" * 80)

    for rank, filt_idx in enumerate(strongest_indices):
        print(
            f"Rank {rank+1}: "
            f"Filter {filt_idx} "
            f"Energy = {activation_energy[filt_idx]:.3e}"
        )

    # -------------------------------------------------------------------------
    # Plot strongest filters
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        len(strongest_indices),
        3,
        figsize=(12, 2.8 * len(strongest_indices)),
    )

    if len(strongest_indices) == 1:
        axes = axes[None, :]

    dt = config.dt

    for row, filt_idx in enumerate(strongest_indices):

        kernel = kernels[filt_idx]

        # ================================================================
        # 1. Time-domain kernel
        # ================================================================

        ax = axes[row, 0]

        ax.plot(kernel, linewidth=1.5)

        ax.set_ylabel(f"Filter {filt_idx}")

        if row == 0:
            ax.set_title("Kernel")

        # ================================================================
        # 2. FFT
        # ================================================================

        ax = axes[row, 1]

        n_fft = 2048

        padded = np.zeros(n_fft)
        padded[:len(kernel)] = kernel

        freqs = np.fft.rfftfreq(n_fft, d=dt) / 1000.0

        fft_mag = np.abs(
            np.fft.rfft(padded)
        )

        if np.max(fft_mag) > 0:
            fft_mag = fft_mag / np.max(fft_mag)

        ax.plot(freqs, fft_mag, linewidth=1.2)

        ax.set_xlim(0, 1000)

        ax.axvspan(
            400,
            507,
            alpha=0.2,
        )

        if row == 0:
            ax.set_title("FFT")

        # ================================================================
        # 3. Mean activation vs time
        # ================================================================

        ax = axes[row, 2]

        acts = activation_store[filt_idx]

        for class_idx, class_name in enumerate(config.classes):

            mask = targets == class_idx

            if np.sum(mask) == 0:
                continue

            mean_act = np.mean(
                acts[mask],
                axis=0,
            )

            ax.plot(
                mean_act,
                linewidth=1.5,
                label=class_name,
            )

        if row == 0:
            ax.set_title("Mean activation")

        if row == len(strongest_indices) - 1:
            ax.legend(frameon=False)

    axes[-1, 0].set_xlabel("Kernel index")
    axes[-1, 1].set_xlabel("Frequency (kHz)")
    axes[-1, 2].set_xlabel("Activation time index")

    fig.suptitle(
        f"Strongest filters: {layer_name}",
        y=1.02,
    )

    fig.tight_layout()

    out_path = os.path.join(
        save_folder,
        f"strongest_filters_{layer_name}.png",
    )

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    wandb.save(out_path)


def plot_first_layer_conv_filters(model, config, save_folder):
    """
    Plot first-layer convolution kernels in time and frequency.
    Assumes model.c1.conv exists.
    """

    conv_layer = model.c1.conv

    weights = conv_layer.weight.detach().cpu().numpy()
    filters = weights[:, 0, :]

    n_filters = filters.shape[0]
    kernel_index = np.arange(filters.shape[1])

    # ---------------------------------------------------------
    # Time-domain filters
    # ---------------------------------------------------------

    n_cols = 8
    n_rows = int(np.ceil(n_filters / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.2 * n_cols, 1.8 * n_rows),
        sharex=True,
        sharey=True,
    )

    axes = np.asarray(axes).ravel()

    for i in range(n_filters):
        ax = axes[i]
        ax.plot(kernel_index, filters[i], linewidth=1.0)
        ax.set_title(f"Filter {i}", fontsize=8)

    for i in range(n_filters, len(axes)):
        axes[i].axis("off")

    fig.supxlabel("Kernel index")
    fig.supylabel("Weight")
    fig.tight_layout()

    time_path = os.path.join(save_folder, "first_layer_filters_time.png")
    fig.savefig(time_path, dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------
    # Frequency-domain filters
    # ---------------------------------------------------------

    n_fft = 2048
    dt = config.dt

    freqs = np.fft.rfftfreq(n_fft, d=dt) / 1000.0

    fig, ax = plt.subplots(figsize=(7, 4))

    for i in range(n_filters):
        padded = np.zeros(n_fft)
        padded[:filters.shape[1]] = filters[i]

        filt_fft = np.abs(np.fft.rfft(padded))

        if np.max(filt_fft) > 0:
            filt_fft = filt_fft / np.max(filt_fft)

        ax.plot(freqs, filt_fft, linewidth=0.8, alpha=0.6)

    ax.set_xlim(0, 1000)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Normalized magnitude")
    ax.set_title("First-layer convolution filter FFTs")

    fig.tight_layout()

    fft_path = os.path.join(save_folder, "first_layer_filters_fft.png")
    fig.savefig(fft_path, dpi=300)
    plt.close(fig)

    wandb.save(time_path)
    wandb.save(fft_path)

        
def holdout(config, device):
    """ Training of model while evaluating on separate validation set """
    print("HOLDOUT **********************************************************")
    # Get training dataset, scaler, and data shapes for model creation
    train, train_scaler, feature_dim, output_dim = get_train_data(config)  
    test,_,_ = get_test_data(config, train_scaler)

    print("Making data loaders...\n")
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True) 
    test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=True)
    
    # Create neural network
    if config.pretrained_model is None:
        model = create_model(config, device, feature_dim, output_dim)
    else:
        model = load_trained_model(config.pretrained_model, config,
                                   feature_dim, output_dim,
                                   'model_holdout.pth')
    
    # Make the loss and optimizer
    loss_func = set_loss_func(config)
    optimizer = set_optimizer(config, model)
    
    # Train 
    train_loss_history, test_loss_history, epochs = train_model(
                                                   config, device,
                                                   model, train_loader, 
                                                   test_loader, 
                                                   loss_func, 
                                                   optimizer)
    
    # Save trained model
    torch.save(model.state_dict(),
               os.path.join(wandb.run.dir,'model_holdout.pth')) 
    wandb.save('model_holdout.pth')
    
    train_loss, train_predicted, train_targets, train_indices,\
        train_class_accuracy, train_total_accuracy, train_metrics = \
                get_model_predictions_classification(config, device, model, 
                                         train_loader, loss_func)
    test_loss, test_predicted, test_targets, test_indices,\
        test_class_accuracy, test_total_accuracy, test_metrics = \
                get_model_predictions_classification(config, device, model, 
                                         test_loader, loss_func)            
                
    # Log results          
    print('Logging holdout results..\n') 
    holdout_results = {'holdout_train_loss_history': train_loss_history,
               'holdout_train_loss': train_loss,
               'holdout_train_predicted': train_predicted,
               'holdout_train_targets': train_targets,
               'holdout_train_indices': train_indices,
               'holdout_train_class_accuracy': train_class_accuracy,
               'holdout_train_total_accuracy': train_total_accuracy,
               'holdout_train_metrics': train_metrics,
               'holdout_test_loss_history': test_loss_history,
               'holdout_test_loss': test_loss,
               'holdout_test_predicted': test_predicted,
               'holdout_test_targets': test_targets,
               'holdout_test_indices': test_indices,
               'holdout_test_class_accuracy': test_class_accuracy,
               'holdout_test_total_accuracy': test_total_accuracy,
               'holdout_test_metrics': test_metrics}
    wandb.log({'holdout_results': holdout_results})
        
  
    plot_train_test_loss_curve(train_loss_history, test_loss_history,
                            epochs, name='Holdout',
                            save_folder = wandb.run.dir)
    wandb.save('loss_curve_holdout.png') 
    
    # Class activation maps
    if config.model_architecture =='NeuralNetwork_Conv1D':
        get_class_activation_maps(model, test_loader, config.classes, config.sig_len,
                                  config.dt, config.feature)

    return


def train_model(config, device, model, train_loader, valid_loader, loss_func,
                optimizer):
    """ Train model with data, loss func, optimizer. Evaluate on valid data """
    train_loss_history, valid_loss_history, epochs = [], [], []
    for epoch in tqdm(range(config.epochs)):  
        epochs.append(epoch) 
        train_loss_epoch = train_epoch(config, device, train_loader, model, 
                                       optimizer, loss_func)
        train_loss_history.append(train_loss_epoch)
        valid_loss_epoch,_,_,_ = evaluate_epoch(config, device, valid_loader, 
                                                model, loss_func)
        valid_loss_history.append(valid_loss_epoch) 
    # Convert datatype to numpy               
    train_loss_history = [t.detach().cpu().numpy() for t in train_loss_history]
    valid_loss_history = [t.detach().cpu().numpy() for t in valid_loss_history]
    
    return train_loss_history, valid_loss_history, epochs


def pretrain_model(config, device, model, train_loader, loss_func, optimizer):
    """ Train model for reconstruction error """
    train_loss_history, epochs = [], [],
    for epoch in tqdm(range(config.epochs)):  
        epochs.append(epoch) 
        train_loss_epoch = train_epoch(config, device, train_loader, model, 
                                       optimizer, loss_func)
        train_loss_history.append(train_loss_epoch)
    # Convert datatype to numpy               
    train_loss_history = [t.detach().cpu().numpy() for t in train_loss_history]
    
    return train_loss_history, epochs


def train_model_with_test(config, device, model, train_loader, valid_loader, 
                          test_loader, loss_func, optimizer):
    """ Train model with data, loss func, optimizer. Evaluate valid & test """
    train_loss_history, valid_loss_history, test_loss_history, epochs = \
        [], [], [], []
    for epoch in tqdm(range(config.epochs)):  
        epochs.append(epoch) 
        train_loss_epoch = train_epoch(config, device, train_loader, model, 
                                       optimizer, loss_func)
        train_loss_history.append(train_loss_epoch)
        valid_loss_epoch,_,_,_ = evaluate_epoch(config, device, valid_loader, 
                                                model, loss_func)
        valid_loss_history.append(valid_loss_epoch)      
        test_loss_epoch,_,_,_ = evaluate_epoch(config, device, test_loader,
                                               model, loss_func)
        test_loss_history.append(test_loss_epoch)
    # Convert datatype to numpy               
    train_loss_history = [t.detach().cpu().numpy() for t in train_loss_history]
    valid_loss_history = [t.detach().cpu().numpy() for t in valid_loss_history]
    test_loss_history = [t.detach().cpu().numpy() for t in test_loss_history]
    
    return train_loss_history, valid_loss_history, test_loss_history, epochs


def train_epoch(config, device, train_loader, model, optimizer, loss_func):
    """ Train model on training data for an epoch, return avg batch loss """
    model.train()
    train_loss_epoch = 0     
    for batch_idx, (batch_features, batch_targets, batch_indices) in\
                                                       enumerate(train_loader):
        batch_loss = train_batch(batch_features, batch_targets, model,
                                 optimizer, loss_func, device)
        train_loss_epoch += batch_loss 
    train_loss_epoch = train_loss_epoch / len(train_loader)
    
    return train_loss_epoch
        

def train_batch(batch_features, batch_targets, model, optimizer, loss_func, 
                device):
    """ Compute batch loss from model prediction, do backprop, take step """
    batch_features, batch_targets = batch_features.to(device),\
                                 batch_targets.to(device) 
    optimizer.zero_grad()
    predicted = model(batch_features) # forward pass
    # predicted = predicted[:,0] # flatten to get rid of 2D (regression)
    batch_loss = loss_func(predicted, batch_targets)
    batch_loss.backward() # backpropagation
    optimizer.step()

    return batch_loss


def evaluate_epoch(config, device, loader, model, loss_func):
    """ Evaluate model on data for an epoch, no backprop only forward pass """
    model.eval() 
    loss_epoch = 0      
    predicted, targets, indices = [], [], []  
    for batch_idx, (batch_features, batch_targets, batch_indices) in \
                                                            enumerate(loader):
        batch_loss, batch_predicted = evaluate_batch(batch_features, 
                                                     batch_targets, model, 
                                                     loss_func, device)
        loss_epoch += batch_loss
        predicted.append(batch_predicted)
        targets.append(batch_targets)
        indices.append(batch_indices)
        
    loss_epoch = loss_epoch / len(loader) # avg batch loss
    # Change datatype and get 1D list
    predicted = [t.cpu().detach().numpy() for t in predicted]
    targets = [t.cpu().detach().numpy() for t in targets]
    indices = [t.cpu().detach().numpy() for t in indices]
    targets = flatten(targets) 
    predicted = flatten(predicted)
    indices = flatten(indices)
    
    return loss_epoch, predicted, targets, indices


def evaluate_batch(batch_features, batch_targets, model, loss_func, device):
    """ Compute model predictions / loss on a batch without backprop """
    batch_features, batch_targets = batch_features.to(device), \
                                 batch_targets.to(device)

    # Forward pass 
    batch_predicted = model(batch_features)
    # batch_predicted = batch_predicted[:,0] # flatten to get rid of 2D (reg)
    batch_loss = loss_func(batch_predicted, batch_targets)

    return batch_loss, batch_predicted   


def get_model_predictions_classification(config, device, model, loader,  
                                         loss_func):
    """ Get model predictions on data, classifier """
    model.eval()
    with torch.no_grad(): # test
        loss, predicted, targets, indices = evaluate_epoch(config, device,
                                                           loader, model, 
                                                           loss_func)
    predictions = np.argmax(predicted,axis=1) 
    targets = np.argmax(targets,axis=1)
    
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in config.classes}
    total_pred = {classname: 0 for classname in config.classes}
    class_accuracy = {classname: 0 for classname in config.classes}
    
    for target, prediction in zip(targets, predictions):
           if target == prediction:
               correct_pred[config.classes[target]] += 1
           total_pred[config.classes[target]] += 1
    
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        class_accuracy[classname] = accuracy
    
    total_correct = (predictions==targets).sum().item()
    total_accuracy = total_correct / len(targets) * 100
    
    # Compute Precision, Recall, Accuracy
    print(classification_report(predictions,targets,
                                target_names=config.classes))
    metrics = classification_report(predictions,targets,
                                target_names=config.classes,
                                    output_dict=True)
    
    return loss, predictions, targets, indices, class_accuracy, \
            total_accuracy, metrics
  

def get_model_predictions_autoencoder(config, device, model, loader,
                                      loss_func):
    model.eval()
    with torch.no_grad(): # test
        loss, predicted, targets, indices = evaluate_epoch(config, device,
                                                           loader, model, 
                                                           loss_func)
        
    return loss, predicted, targets, indices
                
def get_class_activation_maps(model, loader, classes, sig_len, dt, feature):  
    """ Obtains intermediate weights from Conv1D and visualize "
    
        @article{ nanbhas2020forwardhook,
      title   = "Intermediate Activations — the forward hook",
      author  = "Bhaskhar, Nandita",
      journal = "Blog: Roots of my Equation (web.stanford.edu/~nanbhas/blog/)",
      year    = "2020",
      url     = "https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/"
    }
        
    """
    
    activation={}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook    
    
    cams = {} # For storing the upsampled class activation maps
    
    for batch_features, batch_targets, batch_indices in loader:   
        
        h1=model.c2.register_forward_hook(get_activation('c2'))
        output = model(batch_features)
        z=activation['c2']
        z=nn.functional.relu(z)
        weights=model.linear.weight.detach().numpy()
        CAM=np.dot(weights,z)
        CAM = (CAM - CAM.min(axis=2, keepdims=True)) / \
            (CAM.max(axis=2, keepdims=True) - CAM.min(axis=2, keepdims=True))
        c = np.exp(CAM) / np.sum(np.exp(CAM), axis=2, keepdims=True)
        targets = np.argmax(batch_targets,axis=1)
        
        for idx,_ in enumerate(batch_targets):
            ex_idx = batch_indices[idx] # examples index
            
            c_example = c[int(targets[idx]),idx,:].squeeze()
            example = batch_features[idx] # the features (waveform or fft)
            
            # Upsample the class activation map to match signal length
            c_example = np.interp(np.arange(len(example)), np.linspace(0,
                                  len(example), num=len(c_example)), c_example)
            
            # Plot the example and upsampled class activation map
            fig = plt.figure(figsize=(10,5))
            ax = plt.subplot(111)
            font_size = 15
            if feature == 'iota': # Plot Waveform
                duration = sig_len*dt*10**6 
                time = np.linspace(0,duration,sig_len) 
                ax.plot(time,example)
                ax.set_ylabel('Amplitude', fontsize=font_size+5)
                ax.set_xlabel('Time ($\mu$s)', fontsize=font_size+5)
                ax.scatter(time, example, cmap='hot_r', c=c_example, s=10)

            elif feature == 'theta': # Plot Normalized FFT
                w,_ = calc_fft(np.arange(2048), dt=10**-7, low_pass=0,
                               high_pass=120*10**4) # get frequency x-axis
                w = w/1000
                ax.plot(w, example)
                ax.set_xlabel('Freq (khz)', fontsize=font_size+5) 
                ax.set_ylabel('Amplitude', fontsize=font_size+5)
                ax.scatter(w, example, cmap='hot_r', c=c_example, s=10)

            #plt.colorbar(im)
            title = f'Test #: {batch_indices[idx]} | '+ \
                        f'Label: {classes[int(targets[idx])]}'
            ax.set_title(title)
            ax.tick_params(axis='x', labelsize=font_size) 
            ax.tick_params(axis='y', labelsize = font_size)
            ax.tick_params(axis='x', which='minor', bottom=True)
            ax.tick_params(which='major', length=7, width = 1.5) 
            ax.tick_params(which='minor', length=4, width = 1.5) 
            fig.tight_layout() 
            
            # Save image            
            name=f'cam_{batch_indices[idx]}.png'
            save_folder=wandb.run.dir
            path = os.path.join(save_folder, name)
            fig.savefig(path)
            plt.show()
            wandb.save(f'cam_{batch_indices[idx]}.png') 

            # Record the color weights used for plotting
            cams[f'{batch_indices[idx]}'] = list(c_example)

            #if idx == 1:
            #    break
        #break # only look at one batch
    
    # Save upsampled class activation maps from each example
    wandb.log({'cams': cams}) 
    # detach hooks
    h1.remove()
    

    return   
    
