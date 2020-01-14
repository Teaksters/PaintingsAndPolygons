import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


colors = ['#d7191c', '#fdae61', '#abd9e9', '#2c7bb6']


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


if __name__ == "__main__":
    datafp = os.path.join("Results", "HC-DATA.csv")
    df = read_data_file(datafp)
    # Split data per painting
    dfs, paintings = split_df(df, 'Painting')
    for i in range(len(paintings)):
        # Per painting split on total amount of vertices used
        dfs[i] = dfs[i].sort_values(by=['Vertices'])
        dfs_vtot, V_tot = split_df(dfs[i], 'Vertices')
        for j in range(len(V_tot)):
            # Per painting and vertices used split on vertices per polygon
            dfs_vtot[j] = dfs_vtot[j].sort_values(by=['Vertices Per Polygon'])
            dfs_Vp, V_p = split_df(dfs_vtot[j], 'Vertices Per Polygon')

            # Format data to plottable state
            sample_MSE_list = [sample["MSE"].to_list() for sample in dfs_Vp]
            sample_MSE_list = np.array(sample_MSE_list)

            all_MSE.append(sample_MSE_list)

            # calculate bins
            min = np.min(sample_MSE_list.flatten())
            max = np.max(sample_MSE_list.flatten())
            step = (max - min) / 15
            bins = np.arange(min, max, step)
            for i in range(len(sample_MSE_list)):
                plt.hist(sample_MSE_list[i], color=colors[i], alpha=0.3,
                         label=V_p[i], bins=bins)
            plt.legend()
            plt.show()
