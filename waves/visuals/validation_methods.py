



def plot_errors_from_multiple_trials(trials, train_mean_errors,
                                     train_std_errors, valid_mean_errors,
                                     valid_std_errors, 
                                     mean_of_train_mean_errors, 
                                     std_of_train_mean_errors, 
                                     mean_of_valid_mean_errors,
                                     std_of_valid_mean_errors, xlabel='Fold', 
                                     name='fold_errors', title='Kfold', 
                                     save_folder=None):
    # Create figure
    # fig, spec2 = create_figure('',
    #                            columns=1,rows=1,width=8,height=8,
    #                            default_font_size=18,tick_font_size=25,
    #                            legend_font_size=22, axes_font_size=28,
    #                            title_font_size=28)
    fig,spec2 = create_figure('',
                                columns=1,rows=1,width=8,height=8,
                                default_font_size=28,tick_font_size=32,
                                legend_font_size=24, axes_font_size=34,
                                title_font_size=38)
    ax = fig.add_subplot(spec2[0,0]) 
    x = np.arange(len(trials)) # get tick marks
    ax.scatter(x,train_mean_errors,color='blue')
    ax.scatter(x,valid_mean_errors,color='darkorange')
    ax.axhline(mean_of_train_mean_errors,color='blue',linestyle='-',
               label = 'Train | Avg +/- $\sigma$')
    ax.axhline(mean_of_train_mean_errors-std_of_train_mean_errors,color='blue',
               linestyle='--')
    ax.axhline(mean_of_train_mean_errors+std_of_train_mean_errors,color='blue',
               linestyle='--')    
    ax.axhline(mean_of_valid_mean_errors,color='darkorange',linestyle='-',
               label = 'Valid | Avg +/- $\sigma$')
    ax.axhline(mean_of_valid_mean_errors-std_of_valid_mean_errors,color='darkorange',
               linestyle='--')
    ax.axhline(mean_of_valid_mean_errors+std_of_valid_mean_errors,color='darkorange',
               linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(trials)
    ax.legend()
    plt.ylabel('Model Average Error (%)')
    plt.xlabel(xlabel)
    #plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(left=0.25,bottom=0.15)

    plt.title('')
    save_figure(fig, name, save_folder)

    return