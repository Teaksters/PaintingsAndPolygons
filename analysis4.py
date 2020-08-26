import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


afg = ['Bach', 'Dali', 'The Kiss', 'Mona Lisa', 'Mondriaan', 'Convergence', 'Salvator Mundi', 'The Starry Night', 'Dama con l\'ermellino']

def read_data_file(fp):
    '''Reads given datafile format sorts by painting (sorting might be
    redundant)

    params: fp - filepath data
    returns: df - pandas datastructure containing sorted data'''
    df = pd.read_csv(fp)
    return df.sort_values(by=['Painting'])


def split_df(df, column):
    '''Split df into multiple seperate df's based on uniques available in
    provided column.

    params: df - Pandas dataframe
    column - column in given dataframe

    returns: dfs - list of Pandas dataframes resulting after split
    uniques - list of unique values from column corresponding to df in dfs.'''
    uniques = df[column].unique().tolist()
    dfs = []
    for u in uniques:
        dfs.append(df.loc[df[column] == u])
    return dfs, uniques

def minmax_std_mean_plot(V_p, MSE_val, painting):
    '''MSE_vals: array [[vals for 3], ..., [vals for 6]]'''
    y_std = np.std(MSE_val, axis=1)
    y_mean = np.mean(MSE_val, axis=1)
    y_min = np.min(MSE_val, axis=1)
    y_max = np.max(MSE_val, axis=1)
    x = np.array([V_p[i] for i in range(len(MSE_val)) for j in range(len(MSE_val[i]))])
    y = MSE_val.flatten()
    X_uniq = np.unique(x)

    # Plot the scatter and fitted line
    fig = plt.figure()
    plt.plot(X_uniq, y_mean, color='blue', label='mean')  # If including mean
    plt.fill_between(X_uniq, y_mean - y_std, y_mean + y_std, color='blue', alpha=0.5, label='standard deviation')
    plt.fill_between(X_uniq, y_min, y_max, color='blue', alpha=0.2, label='min-max')
    title = painting
    plt.legend(loc='lower right')
    plt.title(title)
    plt.xlabel('Vertices Per Polygon (Vp)')
    plt.ylabel('Normalized Score')

    # set axes range
    plt.yticks(np.arange(0.0, 1.0, 0.1))
    plt.xticks(X_uniq, X_uniq)
    plt.grid(axis='y')
    fig_fp = os.path.join('Analysis2', 'General', 'Painting', painting + '.png')
    plt.savefig(fig_fp)
    plt.close()
    return fig

def multi_plot(data, paintings=afg, V_p=[3, 4, 5, 6]):
    '''MSE_vals: array [[vals for 3], ..., [vals for 6]]'''
    fig, ax = plt.subplots(3, 3, sharex='col', sharey='row',
                           constrained_layout=False, figsize = (8,8))
    plt.setp(ax, xticks=[3, 4, 5, 6], yticks=np.arange(0, 1.1, 0.25))
    for i in range(len(paintings)):
        y_std = np.std(data[i], axis=1)
        y_mean = np.mean(data[i], axis=1)
        y_min = np.min(data[i], axis=1)
        y_max = np.max(data[i], axis=1)
        x = np.array([V_p[j] for j in range(len(data[i])) for k in range(len(data[i][j]))])
        y = data[i].flatten()
        X_uniq = np.unique(x)

        # Plot the scatter and fitted line
        ax[int(i / 3), i % 3].plot(X_uniq, y_mean, color='blue', label='mean')  # If including mean
        ax[int(i / 3), i % 3].fill_between(X_uniq, y_mean - y_std, y_mean + y_std, color='blue', alpha=0.5, label='standard deviation')
        ax[int(i / 3), i % 3].fill_between(X_uniq, y_min, y_max, color='blue', alpha=0.2, label='min-max')
        title = paintings[i]
        ax[int(i / 3), i % 3].grid(axis='y', linestyle='--')
        ax[int(i / 3), i % 3].set_title(title)
        if int(i / 3) == 1 and i % 3 == 0:
            ax[int(i / 3), i % 3].set_ylabel('Normalized Score')
        if int(i / 3) == 2 and i % 3 == 1:
            ax[int(i / 3), i % 3].set_xlabel('Vertices Per Polygon (V_p)')


    # handles, labels = ax[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="center right")
    fig.suptitle('Painting Specific Performance')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_fp = os.path.join('Analysis2', 'General', 'Painting', '9_luik.png')
    fig.savefig(fig_fp)
    plt.close()
    return fig

if __name__=='__main__':
    datafp = os.path.join("Results", "HC-DATA.csv")
    df = read_data_file(datafp)
    print(df.shape)
    data = []
    temp = []
    fin = []
    c = 0
    # Split data per painting
    # df.sort_values(by='Vertices Per Polygon')
    # dfs, split_p = split_df(df, 'Vertices Per Polygon')
    # for df_vp in dfs:
    #     data.append(df_vp['Scaled'].to_numpy())
    # print(np.min(data, axis=1), np.max(data, axis=1))
    # minmax_std_mean_plot([3, 4, 5, 6], np.array(data), 'General Trend Normalized Scores Per Polygon Type')

    dfs, split_p = split_df(df, 'Painting')
    for dfss in dfs:
        dfss = dfss.sort_values(by=['Vertices Per Polygon'])
        dfsss, split_vp = split_df(dfss, 'Vertices Per Polygon')
        for df_vp in dfsss:
            data.append(df_vp['Scaled'].to_numpy())
        data = np.array(data)
        fin.append(data)
        c += 1
        data = []
    multi_plot(fin)
    print('done')
