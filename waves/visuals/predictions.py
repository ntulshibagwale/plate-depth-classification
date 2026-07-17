import matplotlib.pyplot as plt
import numpy as np
from waves.visuals.util import save_figure
from matplotlib import ticker
from mpltools import annotation

def plot_angle_predictions_wrt_targets(targets, predicted, name,
                                       title, save_folder = None):
    plt.style.use('default')

    fig, ax = plt.subplots()

    # cmap = plt.get_cmap('Blues')
    #colors = np.linspace(0.6, 1, 2)
    #colors = cmap(colors)
    x = np.linspace(0,100,5)
    y = np.linspace(0,100,5)
    ax.scatter(targets, predicted, color='black',linewidth=2.5,marker='o')
    ax.plot(x, y, linestyle ='-')  

    plt.ylabel('Predicted ($\\theta$)')
    plt.xlabel('Target ($\\theta$)')

    plt.title(title)
    
    # Make pretty
    MEDIUM_SIZE=15
    ax.set_xlabel('True Angle ($^\circ$)', fontsize=MEDIUM_SIZE+5) # NOTE: axis labels
    ax.set_ylabel('Predicted Angle ($^\circ$)', fontsize=MEDIUM_SIZE+5)
    ax.tick_params(axis='x', labelsize=MEDIUM_SIZE) # NOTE: Size of tick marks on x axis
    ax.tick_params(axis='y', labelsize = MEDIUM_SIZE) # NOTE: Size of tick marks on y axis
    ax.tick_params(axis='x', which='minor', bottom=True)
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(2.5)) # NOTE: Interval of major tick marks on y axis
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5)) # NOTE: interval of major tick marks on x axis
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) # NOTE: Interval of major tick marks on y axis
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(5)) # NOTE: interval of major tick marks on x axis
    ax.set_xlim([15, 45])
    ax.set_ylim([15, 45])
    ax.tick_params(which='major', length=7, width = 1.5) # NOTE: size and width of major tick marks
    ax.tick_params(which='minor', length=4, width = 1.5) # NOTE: size and width of minor tick marks
    #plt.legend(fontsize=MEDIUM_SIZE,loc='upper right') # NOTE: Prints legend with specified font size
    fig.tight_layout() # NOTE: prevents clipping of plot
    
    annotation.slope_marker((31, 30.75), (1, 1), ax=ax,
                            text_kwargs={'color': 'black'},
                            poly_kwargs={'facecolor': (0.73, 0.8, 1)})
    
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box') # NOTE: makes plot square

    save_figure(fig, name, save_folder)


    return 

def plot_residuals(targets, predicted, name, title, save_folder = None):
    plt.style.use('default')

    fig, ax = plt.subplots()

    # cmap = plt.get_cmap('Blues')
    #colors = np.linspace(0.6, 1, 2)
    #colors = cmap(colors)
    x = np.linspace(0,100,5)
    y = np.zeros(len(x))
    res = []
    for idx, _ in enumerate(targets):
        res.append(predicted[idx]-targets[idx])
     
    ax.scatter(targets, res, color='black',linewidth=2.5,marker='o')
    ax.plot(x, y, linestyle ='-')  

    plt.ylabel('Predicted ($\\theta$)')
    plt.xlabel('Target ($\\theta$)')

    plt.title(title)
    
    # Make pretty
    MEDIUM_SIZE=15
    ax.set_xlabel('True Angle ($^\circ$)', fontsize=MEDIUM_SIZE+5) # NOTE: axis labels
    ax.set_ylabel('Residuals ($^\circ$)', fontsize=MEDIUM_SIZE+5)
    ax.tick_params(axis='x', labelsize=MEDIUM_SIZE) # NOTE: Size of tick marks on x axis
    ax.tick_params(axis='y', labelsize = MEDIUM_SIZE) # NOTE: Size of tick marks on y axis
    ax.tick_params(axis='x', which='minor', bottom=True)
   # ax.yaxis.set_minor_locator(ticker.MultipleLocator(2.5)) # NOTE: Interval of major tick marks on y axis
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5)) # NOTE: interval of major tick marks on x axis
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) # NOTE: Interval of major tick marks on y axis
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(5)) # NOTE: interval of major tick marks on x axis
    ax.set_xlim([15, 45])
    #ax.set_ylim([15, 45])
    ax.tick_params(which='major', length=7, width = 1.5) # NOTE: size and width of major tick marks
    ax.tick_params(which='minor', length=4, width = 1.5) # NOTE: size and width of minor tick marks
    #plt.legend(fontsize=MEDIUM_SIZE,loc='upper right') # NOTE: Prints legend with specified font size
    fig.tight_layout() # NOTE: prevents clipping of plot
        
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box') # NOTE: makes plot square

    save_figure(fig, name, save_folder)


    return 