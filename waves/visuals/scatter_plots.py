import matplotlib.pyplot as plt
import numpy as np
from waves.visuals.util import create_figure
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker

def plot_scatter_multi_color(means, stds, classes, colors, ylabel, save_path,
                             y_tick_spacing=0.1):
    """ Plotting scatter plot, each x has a different color """
    fig = plt.figure(figsize=(5,5))
    font_size=15
    ax = plt.subplot(111)
    x = np.arange(len(classes))
    ax.scatter(x,means,color=colors)
    for pos, y, err, col in zip(x, means, stds, colors):
        ax.errorbar(pos, y, err, lw=2, capsize=4, capthick=4, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel(ylabel, fontsize=font_size+2)
    ax.tick_params(axis='x', labelsize=font_size) 
    ax.tick_params(axis='y', labelsize = font_size) 
    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_spacing)) 
    ax.tick_params(which='major', length=7, width = 1.5) 
    ax.tick_params(which='minor', length=4, width = 1.5) 
    ax.set_xticks(x)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box') 
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return 

def plot_scatter_2_sensors(means, stds, classes, ylabel, save_path):
    # Plotting scatter plot for arbitrary number of classes
    # with mean and error bars
    fig = plt.figure(figsize=(10, 15))
    font_size=15
    ax = plt.subplot(111)
    x = np.arange(len(classes))
    sensor1_means = []
    sensor2_means = []
    sensor1_stds = []
    sensor2_stds = []
    for idx,_ in enumerate(means):
        sensor1_means.append(means[idx][0])
        sensor2_means.append(means[idx][1])
        sensor1_stds.append(stds[idx][0])
        sensor2_stds.append(stds[idx][1])
        
    ax.scatter(x,sensor1_means)
    ax.errorbar(x,sensor1_means, sensor2_stds, ls='none',lw = 2,
                    capsize = 4, capthick = 4, label='Sensor 1')
    ax.scatter(x,sensor2_means)
    ax.errorbar(x,sensor2_means, sensor2_stds, ls='none',lw = 2,
                    capsize = 4, capthick = 4, label='Sensor 2')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel(ylabel, fontsize=font_size+2)
    ax.tick_params(axis='x', labelsize=font_size) 
    ax.tick_params(axis='y', labelsize = font_size) 
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10)) 
    ax.tick_params(which='major', length=7, width = 1.5) 
    ax.tick_params(which='minor', length=4, width = 1.5) 
    ax.set_xticks(x)
    ax.legend(fontsize=font_size, loc ='upper right')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box') 
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return 

def plot_scatter(means, stds, classes, ylabel, xlabel, save_path):
    # Plotting scatter plot for arbitrary number of classes
    # with mean and error bars
    fig = plt.figure(figsize=(6, 6))
    font_size=18
    ax = plt.subplot(111)
    x = np.arange(len(classes))
    ax.scatter(x,means)
    ax.errorbar(x,means, stds, ls='none',lw = 2,
                    capsize = 4, capthick = 4)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_ylabel(ylabel, fontsize=font_size+2)
    ax.set_xlabel(xlabel, fontsize=font_size+2)

    ax.tick_params(axis='x', labelsize=font_size) 
    ax.tick_params(axis='y', labelsize = font_size) 
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20)) 
    ax.tick_params(which='major', length=7, width = 1.5) 
    ax.tick_params(which='minor', length=4, width = 1.5) 
    ax.set_xticks(x)
    #ax.legend(fontsize=font_size, loc ='upper right')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box') 
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return 