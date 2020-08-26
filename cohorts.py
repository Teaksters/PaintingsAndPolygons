import numpy as np
import pandas as pd
import os
from collections import Counter
import copy
import matplotlib.pyplot as plt


colors = ['#ca0020', '#f4a582', '#92c5de', '#0571b0']

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


def main():
    c_min = Counter()
    c_mean = Counter()
    datafp = os.path.join("Results", "HC-DATA.csv")
    df = read_data_file(datafp)
    df.sort_values(by=['Painting', 'Vertices', 'Vertices Per Polygon'],
                   inplace=True)
    dfs, paintings = split_df(df, 'Painting')
    for df_t1 in dfs:
        dfs_t1, vertices = split_df(df_t1, 'Vertices')
        for df_t2 in dfs_t1:
            cohort, vp = split_df(df_t2, 'Vertices Per Polygon')
            c_minlist = []
            c_meanlist = []
            for i in range(len(cohort)):
                c_minlist.append(cohort[i]['Scaled'].min())
                c_meanlist.append(cohort[i]['Scaled'].mean())
            _, c_vp1 = zip(*sorted(zip(c_minlist, copy.deepcopy(vp))))
            _, c_vp2 = zip(*sorted(zip(c_meanlist, copy.deepcopy(vp))))
            c_min[' < '.join(str(v) for v in c_vp1)] += 1
            c_mean[' < '.join(str(v) for v in c_vp2)] += 1

    # total = sum(c_min.values(), 0.0)
    # for key in c_min:
    #     c_min[key] = round((c_min[key] / total) * 100, 1)
    #
    # total = sum(c_mean.values(), 0.0)
    # for key in c_mean:
    #     c_mean[key] = round((c_mean[key] / total) * 100, 1)
    mean_keys = [key for key in c_mean]
    mean_vals = [c_mean[key] for key in mean_keys]
    mean_vals = mean_vals[:3] + [sum(mean_vals[3:])]
    mean_keys = mean_keys[:3] + ['other']
    mean_keys, mean_vals = zip(*sorted(zip(mean_keys, mean_vals)))
    patches = plt.pie(mean_vals, autopct='%1.1f%%', colors=colors)
    plt.legend(patches[0], mean_keys, loc="lower left")
    plt.title('Cohort Percentile Mean Score Comparison')
    plt.savefig('pie_char_mean.png')
    plt.close()

    min_keys = [key for key in c_min]
    min_vals = [c_min[key] for key in min_keys]
    min_vals = min_vals[:4] + [sum(min_vals[4:])]
    min_keys = min_keys[:4] + ['other']
    min_keys, min_vals = zip(*sorted(zip(min_keys, min_vals)))
    patches = plt.pie(min_vals, autopct='%1.1f%%')
    plt.legend(patches[0], min_keys, loc="upper left")
    plt.title('Cohort Percentile Best Score Comparison')
    plt.savefig('pie_char_min.png')


if __name__=='__main__':
    main()
