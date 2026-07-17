import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
import numpy as np


def create_figure(suptitle, columns, rows, width=20, height=10,
                  suptitle_font_size=24, default_font_size=10,
                  title_font_size=12, axes_font_size=12, tick_font_size=10,
                  legend_font_size=10, w_space=0.25, h_space=0.25, lines_lw=1,
                  axes_lw=1.5):
    """
    
    Create a gridspec figure, so more flexibility with subplots.

    Parameters
    ----------
    suptitle : string
        Master title.
    columns : int
        Subplot columns.
    rows : int
        Subplot rows.
    width : int, optional
        Figure width. The default is 20.
    height : int, optional
        Figure height. The default is 10.
    suptitle_font_size : int, optional
        Master title size. The default is 24.
    default_font_size : TYPE, optional
        The default is 10.
    title_font_size : int, optional
        Individual subplot title size. The default is 12.
    axes_font_size : int, optional
        Font size for x and y labels. The default is 12.
    tick_font_size : int, optional
        The default is 10.
    legend_font_size : int, optional
        The default is 10.
    w_space : float, optional
        Distance between subplots horizontally. The default is 0.25.
    h_space : float, optional
        Distance between subplots vertically. The default is 0.25.
    lines_lw : float, optional
        Plot line width. The default is 1.
    axes_lw : float, optional
        Axes line width. The default is 1.        

    Returns
    -------
    fig : Matplotlib object
        The figure handle.
    spec2 : Matplotlib object
        Used for adding custom sized subplots ; fig.add_subplot(spec2[0,0]).

    """
    fig = plt.figure(figsize=(width,height))
    
    # Create subplot grid -> used for subplots
    spec2 = gridspec.GridSpec(ncols = columns, nrows = rows, figure = fig,
                              wspace = w_space,hspace = h_space)
    
    # Master Figure Title
    fig.suptitle(suptitle,fontsize=suptitle_font_size)
    
    # General plotting defaults    
    plt.rc('font', size=default_font_size)     # controls default text size
    plt.rc('axes', titlesize=title_font_size)  # fontsize of the title
    plt.rc('axes', labelsize=axes_font_size)   # fontsize of the x and y labels
    plt.rc('xtick', labelsize=tick_font_size)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=tick_font_size)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=legend_font_size)# fontsize of the legend
    plt.rc('lines', linewidth = lines_lw)
    plt.rc('axes', linewidth = axes_lw) 

    return fig, spec2


def get_location_colors(loc,lower=0.4,upper=1,num=5):
    """
    
    Function to get colors used for different PLB sources.

    Parameters
    ----------
    loc : str
        PLB source: 'top', 'sid', 'bot'
    lower : float, optional
        Lower threshold within color map. The default is 0.4.
    upper : TYPE, optional
        Upper threshold within color map. The default is 1.
    num : TYPE, optional
        Number of divisions between lower and upper. The default is 5.

    Returns
    -------
    colors : array-like
        Array of rgb colors, with size [num x 4].
    cmap_name : str
        Corresponding to cmap: 'top'='Reds' ; 'sid'='Blues' ; 'bot'='Greens'
    name : str
        Capitalized name of PLB source: 'Top', 'Side', 'Bottom'

    """
    if loc == 'top':
        name = 'Top'
        cmap_name = 'Reds'
        cmap = plt.get_cmap(cmap_name)
        colors = np.linspace(lower, upper, num)
        colors = cmap(colors)
    elif loc == 'sid':
        name = 'Side'
        cmap_name = 'Blues'
        cmap = plt.get_cmap(cmap_name)
        colors = np.linspace(lower, upper, num)
        colors = cmap(colors)
    elif loc == 'bot':
        name = 'Bottom'
        cmap_name = 'Greens'
        cmap = plt.get_cmap(cmap_name)
        colors = np.linspace(lower, upper, num)
        colors = cmap(colors)
        
    return colors, cmap_name, name


def save_figure(fig, file_name, save_folder=None):
    """
    
    Save passed matplotlib object to appropriate folder.

    Parameters
    ----------
    fig : matplotlib object
        Created during plotting.
    file_name : str
        File name for saved .png file.
    save_folder : str, optional
        Path for desired save folder. The default is None, saving to work dir.

    """
    if save_folder is not None:
        path = os.path.join(save_folder, file_name)
        fig.savefig(path)
    else: # save_folder == None:
        fig.savefig(file_name)
    plt.show()
    
    return