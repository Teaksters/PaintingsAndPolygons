import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


colors = ['#d7191c', '#fdae61', '#abd9e9', '#2c7bb6']


def prepare_data(file):
    samples = []
    for folder in os.listdir(file):
        x, y, z = folder.split('_')[:3]
        samples.append([x, int(y), int(z), folder])
    return samples


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


def retrieve_MSE_course(resdir, df):
    '''Retrieves the MSE score course during a run of the folder given
    through reading it from the names allotted to the results images.'''
    fp = []
    MSE = []
    iter = []
    for dirname in df['dir']:
        fp = resdir / dirname
        tMSE = []
        titer = []
        for item in os.listdir(fp):
            if item[-4:] == '.txt':
                _, i, m = item[:-4].split('-')
                tMSE.append(int(m))
                titer.append(int(i))
        titer, tMSE = zip(*sorted(zip(titer, tMSE)))
        MSE.append(tMSE)
        iter.append(titer)
    return MSE, iter


if __name__ == "__main__":
    res_file = Path('../Results')
    paintings = [x for x in os.listdir(res_file)
                 if os.path.isdir(res_file / x)]
    for p in paintings:
        cur_res_file = Path(res_file / p)

        # Prepare data and put into pandas datastructure
        samples = prepare_data(cur_res_file)
        labels = ['Painting', 'V_p', 'V_tot', 'dir']
        df = pd.DataFrame(samples, columns=labels)
        df.sort_values(by=['V_p', 'V_tot'], inplace=True)

        # start seperating data for graph plotting
        dfs, _ = split_df(df, 'V_tot')
        for df in dfs:
            dfs_Vp, V_p = split_df(df, 'V_p')
            for df_vp in dfs_Vp:
                MSE, iter = retrieve_MSE_course(cur_res_file, df_vp)
                label = str(list(df_vp['V_p'])[0])
                print(df_vp, label)
                for i in range(len(MSE)):
                    plt.plot(iter[i], MSE[i], label=label, color=colors[int(label) - 3])

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.show()
            plt.close()

            # figure out what to do to middle the values for inter v_p comparison
            # plot figure

    # MSE_course()
