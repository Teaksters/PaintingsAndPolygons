from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
import numpy as np
import pandas as pd
import os
from pathlib import Path
import csv
import itertools


stat_path = Path('Analysis/Statistics/Mann_Whitneyu')


def read_data_file(fp):
    '''Reads given datafile format sorts by painting (sorting might be
    redundant)

    params: fp - filepath data
    returns: df - pandas datastructure containing sorted data'''
    df = pd.read_csv(fp)
    df.drop(['Replication'], axis=1, inplace=True)
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
    scaled = [[], [], [], []]

    combination = [x for x in itertools.combinations(np.arange(4), 2)]
    temp = [str(x[0] + 3) + '-' + str(x[1] + 3) for x in combination]
    # csv_name = 'MannWhitneyu'
    # with open(stat_path / csv_name, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['all data is written in format (statistics, p-value)'])
    #     header = ['Vertices', 'statistics']
    #     header.extend(temp)
    #     print(header)
    #     writer.writerow(header)
    datafp = os.path.join("Results", "HC-DATA.csv")
    df = read_data_file(datafp)
    v_c = 0
    dfspp, vertices = split_df(df, 'Painting')
    vertices, dfspp = zip(*sorted(zip(vertices, dfspp)))
    for dfsp in dfspp:
        csv_name = 'MannWhitneyu'
        p_c = 0
        dfs, paintings = split_df(dfsp, 'Vertices')
        paintings, dfs = zip(*sorted(zip(paintings, dfs)))
        csv_name += vertices[v_c] + '.csv'
        with open(stat_path / csv_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['all data is written in format (statistics, p-value)'])
            header = ['Vertices', 'statistics']
            header.extend(temp)
            print(header)
            writer.writerow(header)
            # stat, p = kruskal(scaled[0], scaled[1], scaled[2], scaled[3])

            for dftje in dfs:
                dftjes, uniquetjes = split_df(dftje, 'Vertices Per Polygon')
                uniquetjes, dftjes = zip(*sorted(zip(uniquetjes, dftjes)))
                w, x, y, z = np.array(dftjes[0]['MSE'].to_list()),\
                             np.array(dftjes[1]['MSE'].to_list()),\
                             np.array(dftjes[2]['MSE'].to_list()),\
                             np.array(dftjes[3]['MSE'].to_list())

                # Scale for generalized statistics
                all_min = min(min(w), min(x), min(y), min(z))
                scaled[0].append(w / all_min)
                scaled[1].append(x / all_min)
                scaled[2].append(y / all_min)
                scaled[3].append(z / all_min)
                scaled = np.array(scaled).reshape((4, -1))

                pstats = []
                for pair in combination:
                    stat, p = mannwhitneyu(scaled[pair[0]], scaled[pair[1]])
                    pstats.append((stat, p))
                row = [paintings[p_c]]
                row.extend(pstats)
                writer.writerow(row)
                scaled = [[], [], [], []]
                p_c += 1
        v_c += 1



if __name__=="__main__":
    main()
