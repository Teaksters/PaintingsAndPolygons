import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Parameters to set the plot type
singular = False
multiple_V = True
multiple_P = False


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


def simple_error_plot(x, y, yerr, painting, V_tot, multiple_V=False,
                      multiple_P=False):
    '''Plots graph'''
    # compute plottables
    if multiple_V:
        line_label = str(V_tot) + " Vertices"
    if multiple_P:
        line_label = painting
    if multiple_P or multiple_V:
        plt.errorbar(x, y, yerr=yerr, label=line_label, marker='o')
    else:
        plt.errorbar(x, y, yerr=yerr, marker='o')
    plt.xticks(x, x)


def singular_plot(V_p, MSE_scores, MSE_err, painting, V_tot):
    # Plot and store graph results single painting single V_tot
    fig = plt.figure()
    simple_error_plot(V_p, MSE_scores, MSE_err, painting, V_tot)
    title = painting + '_total vertices_' + str(V_tot)
    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel('Vertices Per Polygon')
    plt.ylabel('% Offset of mean MSE')
    fig_fp = os.path.join('Analysis', 'singular',
                          painting + "_V_" + str(V_tot) + '.png')
    plt.savefig(fig_fp)
    plt.close()
    return fig


def multiple_V_plot(V_p, MSE_means, MSE_errs, painting, V_tots):
    fig = plt.figure()
    for i in range(len(V_tots)):
        simple_error_plot(V_p, MSE_means[i], MSE_errs[i], painting, V_tots[i],
                          True)
    title = painting + " All Vertice Combinations"
    plt.title(title)
    plt.legend(loc='upper left')
    plt.xlabel('Vertices Per Polygon')
    plt.ylabel('% Offset of mean MSE')
    fig_fp = os.path.join('Analysis', 'vertices', painting + "_All_Vertici.png")
    plt.savefig(fig_fp)
    plt.close()
    return fig


def multiple_P_plot(V_p, MSE_means, MSE_errs, paintings, V_tots):
    fig = plt.figure()
    offset = len(paintings) - 1
    for i in range(len(V_tots)):
        for j in range(len(paintings)):
            simple_error_plot(V_p,
                              MSE_means[offset * j + i],
                              MSE_errs[offset * j + i],
                              paintings[j],
                              V_tots[i], multiple_P=True)
        title = str(V_tots[i]) + " Vertices All Paintings Comparison"
        plt.title(title)
        plt.legend(loc='upper left')
        plt.xlabel('Vertices Per Polygon')
        plt.ylabel('% Offset of mean MSE')
        fig_fp = os.path.join('Analysis', 'paintings',
                              str(V_tots[i]) + "_V_All_Paintings.png")
        plt.savefig(fig_fp)
        plt.close()
    return fig

if __name__ == "__main__":
    multiple_means = []
    multiple_std = []
    V_p = []
    V_tot = []
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

            # Scale for comparison purposes
            all_mean = np.mean(sample_MSE_list)
            MSE_scaled = (sample_MSE_list - all_mean) / all_mean

            # calculate mean and standard deviation
            MSE_std = np.std(MSE_scaled, axis=1)
            MSE_mean = np.mean(MSE_scaled, axis=1)

            if singular:
                singular_plot(V_p, MSE_mean, MSE_std, paintings[i], V_tot[j])

            if multiple_P or multiple_V:
                multiple_means.append(MSE_mean)
                multiple_std.append(MSE_std)

        # Plot multiple V_tot per painting
        if multiple_V:
            multiple_V_plot(V_p, multiple_means, multiple_std, paintings[i],
                            V_tot)
            multiple_means = []
            multiple_std = []

    if multiple_P:
        multiple_P_plot(V_p, multiple_means, multiple_std, paintings,
                        V_tot)
        multiple_means = []
        multiple_std = []






        # Plot and store graph results single painting multiple V_tot
    # plot and store graph showing results all paintings, V_tot & V_p
