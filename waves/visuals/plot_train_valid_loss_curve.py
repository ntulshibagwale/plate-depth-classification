import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from waves.visuals.util import save_figure
import matplotlib
import numpy as np

#matplotlib.rcParams.update(matplotlib.rcParamsDefault)

def plot_train_test_loss_curve(train_loss, test_loss, epochs, name='', 
                                save_folder=None):
    """
    
    Plots train, valid loss curve over training. Saves figure.

    Parameters
    ----------
    train_loss : list
        Training loss over course of training.
        Example: 
            train_loss = [100,80,75,80,64,...]
    test_loss : list
        Test loss over course of training.
    epochs : list
        Number of epochs passed at each train_loss log. Same size as loss.
        Example:
            epochs = [10,20,30,40,....]
    name : str
        Name appended to end of file name. i.e. name = 01
        Figure will be saved as 'loss_curve_01', otherwise just 'loss_curve'

    """
    plt.style.use('default')

    cmap = plt.get_cmap('tab10')
    colors = np.linspace(0, 0.1, 2) # blue and orange
    colors = cmap(colors)

    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss, color=colors[0], linewidth=1.6, label='Train')
    ax.plot(epochs, test_loss, linewidth=1.6,  color=colors[1],
            label='Test')
    
    # Adjust limits if needed for higher resolution in loss curve behavior
    if len(epochs) > 200:
        plt.gca().set_xlim(left=50) # to make loss more clear
        max_y = 0
        if max(train_loss[50:]) > max(test_loss[50:]):
            max_y = max(train_loss[50:])
        else:
            max_y = max(test_loss[50:])
        plt.gca().set_ylim(top = max_y + 40)
        plt.gca().set_ylim(bottom = -10)

    # Make pretty
    MEDIUM_SIZE=15
    ax.set_xlabel('Epoch', fontsize=MEDIUM_SIZE+5) # NOTE: axis labels
    ax.set_ylabel('Loss', fontsize=MEDIUM_SIZE+5)
    ax.tick_params(axis='x', labelsize=MEDIUM_SIZE) # NOTE: Size of tick marks on x axis
    ax.tick_params(axis='y', labelsize = MEDIUM_SIZE) # NOTE: Size of tick marks on y axis
    ax.tick_params(axis='x', which='minor', bottom=True)
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(25)) # NOTE: Interval of major tick marks on y axis
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(125)) # NOTE: interval of major tick marks on x axis
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50)) # NOTE: Interval of major tick marks on y axis
    ax.xaxis.set_major_locator(ticker.MultipleLocator(250)) # NOTE: interval of major tick marks on x axis
    ax.tick_params(which='major', length=7, width = 1.5) # NOTE: size and width of major tick marks
    ax.tick_params(which='minor', length=4, width = 1.5) # NOTE: size and width of minor tick marks
    fig.tight_layout() # NOTE: prevents clipping of plot
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box') # NOTE: makes plot square   
    plt.legend(fontsize=MEDIUM_SIZE,loc='upper right') # NOTE: Prints legend with specified font size

    save_figure(fig,'loss_curve_'+ name, save_folder)

    return

def plot_train_valid_loss_curve(train_loss, valid_loss, epochs, name='', 
                                save_folder=None):
    """
    
    Plots train, valid loss curve over training. Saves figure.

    Parameters
    ----------
    train_loss : list
        Training loss over course of training.
        Example: 
            train_loss = [100,80,75,80,64,...]
    valid_loss : list
        Valid loss over course of training.
    epochs : list
        Number of epochs passed at each train_loss log. Same size as loss.
        Example:
            epochs = [10,20,30,40,....]
    name : str
        Name appended to end of file name. i.e. name = 01
        Figure will be saved as 'loss_curve_01', otherwise just 'loss_curve'

    """
    plt.style.use('default')

    cmap = plt.get_cmap('tab10')
    colors = np.linspace(0, 0.1, 2) # blue and orange
    colors = cmap(colors)

    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss, color=colors[0], linewidth=1.6, label='Train')
    ax.plot(epochs, valid_loss, linewidth=1.6,  color=colors[1],
            label='Valid')
    
    # Adjust limits if needed for higher resolution in loss curve behavior
    if len(epochs) > 200:
        plt.gca().set_xlim(left=50) # to make loss more clear
        max_y = 0
        if max(train_loss[50:]) > max(valid_loss[50:]):
            max_y = max(train_loss[50:])
        else:
            max_y = max(valid_loss[50:])
        plt.gca().set_ylim(top = max_y + 40)
        plt.gca().set_ylim(bottom = -10)

    # Make pretty
    MEDIUM_SIZE=15
    ax.set_xlabel('Epoch', fontsize=MEDIUM_SIZE+5) # NOTE: axis labels
    ax.set_ylabel('Loss', fontsize=MEDIUM_SIZE+5)
    ax.tick_params(axis='x', labelsize=MEDIUM_SIZE) # NOTE: Size of tick marks on x axis
    ax.tick_params(axis='y', labelsize = MEDIUM_SIZE) # NOTE: Size of tick marks on y axis
    ax.tick_params(axis='x', which='minor', bottom=True)
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(25)) # NOTE: Interval of major tick marks on y axis
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(125)) # NOTE: interval of major tick marks on x axis
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50)) # NOTE: Interval of major tick marks on y axis
    ax.xaxis.set_major_locator(ticker.MultipleLocator(250)) # NOTE: interval of major tick marks on x axis
    ax.tick_params(which='major', length=7, width = 1.5) # NOTE: size and width of major tick marks
    ax.tick_params(which='minor', length=4, width = 1.5) # NOTE: size and width of minor tick marks
    fig.tight_layout() # NOTE: prevents clipping of plot
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box') # NOTE: makes plot square   
    plt.legend(fontsize=MEDIUM_SIZE,loc='upper right') # NOTE: Prints legend with specified font size

    save_figure(fig,'loss_curve_'+ name, save_folder)

    return


def plot_train_valid_test_loss_curve(train_loss, valid_loss, test_loss, epochs, name='', 
                                save_folder=None):
    """
    
    Plots train, valid loss curve over training. Saves figure.

    Parameters
    ----------
    train_loss : list
        Training loss over course of training.
        Example: 
            train_loss = [100,80,75,80,64,...]
    valid_loss : list
        Valid loss over course of training.
    epochs : list
        Number of epochs passed at each train_loss log. Same size as loss.
        Example:
            epochs = [10,20,30,40,....]
    name : str
        Name appended to end of file name. i.e. name = 01
        Figure will be saved as 'loss_curve_01', otherwise just 'loss_curve'

    """
    plt.style.use('default')

    cmap = plt.get_cmap('tab10')
    colors = np.linspace(0, 0.2, 3) # blue and orange
    colors = cmap(colors)

    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss, color=colors[0], linewidth=1.6, label='Train')
    ax.plot(epochs, valid_loss, linewidth=1.6,  color=colors[1],
            label='Valid')
    ax.plot(epochs, test_loss, linewidth=1.6, color= colors[2],label='Test')
    
    # Adjust limits if needed for higher resolution in loss curve behavior
    if len(epochs) > 200:
        plt.gca().set_xlim(left=50) # to make loss more clear
        max_y = 0
        if max(train_loss[50:]) > max(valid_loss[50:]):
            max_y = max(train_loss[50:])
        else:
            max_y = max(valid_loss[50:])
        plt.gca().set_ylim(top = max_y + 40)
        plt.gca().set_ylim(bottom = -10)

    # Make pretty
    MEDIUM_SIZE=15
    ax.set_xlabel('Epoch', fontsize=MEDIUM_SIZE+5) # NOTE: axis labels
    ax.set_ylabel('Loss', fontsize=MEDIUM_SIZE+5)
    # ax.tick_params(axis='x', labelsize=MEDIUM_SIZE) # NOTE: Size of tick marks on x axis
    # ax.tick_params(axis='y', labelsize = MEDIUM_SIZE) # NOTE: Size of tick marks on y axis
    # ax.tick_params(axis='x', which='minor', bottom=True)
    # #ax.yaxis.set_minor_locator(ticker.MultipleLocator(25)) # NOTE: Interval of major tick marks on y axis
    # #ax.xaxis.set_minor_locator(ticker.MultipleLocator(125)) # NOTE: interval of major tick marks on x axis
    # #ax.yaxis.set_major_locator(ticker.MultipleLocator(50)) # NOTE: Interval of major tick marks on y axis
    # #ax.xaxis.set_major_locator(ticker.MultipleLocator(250)) # NOTE: interval of major tick marks on x axis
    # ax.tick_params(which='major', length=7, width = 1.5) # NOTE: size and width of major tick marks
    # ax.tick_params(which='minor', length=4, width = 1.5) # NOTE: size and width of minor tick marks
    fig.tight_layout() # NOTE: prevents clipping of plot
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box') # NOTE: makes plot square   
    plt.legend(fontsize=MEDIUM_SIZE,loc='upper right') # NOTE: Prints legend with specified font size

    save_figure(fig,'loss_curve_'+ name, save_folder)

    return