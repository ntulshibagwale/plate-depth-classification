import matplotlib.pyplot as plt
import os

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