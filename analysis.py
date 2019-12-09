import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Parameters to set the plot type
N_paintings = 9
singular = False
multiple_V = False
multiple_P = False
scatter_S = False
scatter_M_gen = True
scatter_M_Vgen = False


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


def max_method(y):
    '''Scales data using the max method'''
    return y / max(y.flatten())


###############################################################
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
    fig_fp = os.path.join('Analysis', 'singular', 'Mean',
                          painting + "_V_" + str(V_tot) + '.png')
    plt.savefig(fig_fp)
    plt.close()
    return fig

def scatter_plot(x, y, label):
    plt.scatter(x, y, marker='x', alpha=0.5, label=label)
    plt.xticks(x, x)


def singular_fitted_scatter_plot(V_p, MSE_val, painting, V_tot):
    # Reorganize data for plotting
    x = np.array([V_p[i] for i in range(len(MSE_val)) for j in range(len(MSE_val[i]))])
    y = np.array(MSE_val).flatten()

    # Create a regression line model of data
    X, Y = x.reshape(-1, 1), y.reshape(-1, 1)
    regr = LinearRegression()
    regr.fit(X, Y)
    X_uniq = np.unique(x).reshape(-1, 1)
    Y_pred = regr.predict(X_uniq)
    X_uniq, Y_pred = X_uniq.flatten(), Y_pred.flatten()

    # Plot the scatter and fitted line
    fig = plt.figure()
    scatter_plot(x, y, 'Data points')
    plt.plot(X_uniq, Y_pred, color='k', label='trend line')
    title = painting + '_total vertices_' + str(V_tot)
    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel('Vertices Per Polygon')
    plt.ylabel('% Offset of mean MSE')
    fig_fp = os.path.join('Analysis', 'singular', 'scatter_Lfit',
                          painting + "_V_" + str(V_tot) + '.png')
    plt.savefig(fig_fp)
    plt.close()
    return fig
#######################################################################


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


def multiple_scatterfit_gen_plot(V_p, MSE_vals, painting):
    # Reorganize data for plotting
    MSE_val = np.array([[MSE_vals[i][j] for i in range(len(MSE_vals))]
                        for j in range(4)])
    MSE_val = MSE_val.reshape(4, -1)
    y_std = np.std(MSE_val, axis=1)
    y_mean = np.mean(MSE_val, axis=1)
    x = np.array([V_p[i] for i in range(len(MSE_val)) for j in range(len(MSE_val[i]))])
    y = MSE_val.flatten()

    # Create a regression line model of data
    X, Y = x.reshape(-1, 1), y.reshape(-1, 1)
    regr = LinearRegression()
    regr.fit(X, Y)
    X_uniq = np.unique(x).reshape(-1, 1)
    Y_pred = regr.predict(X_uniq)
    X_uniq, Y_pred = X_uniq.flatten(), Y_pred.flatten()

    # Plot the scatter and fitted line
    fig = plt.figure()
    scatter_plot(x, y, 'Data points')
    plt.plot(X_uniq, Y_pred, color='k', label='trend line')
    # plt.plot(X_uniq, y_mean, color='blue', label='mean')  # If including mean
    plt.fill_between(X_uniq, y_mean - y_std, y_mean + y_std, color='blue', alpha=0.3, label='standard deviation')
    title = 'All_paintings_General_trend_V'
    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel('Vertices Per Polygon')
    plt.ylabel('MSE / max_MSE')

    # set axes range
    # plt.yticks(np.arange(0.0, 1.0, 0.1))
    plt.grid(axis='y')
    fig_fp = os.path.join('Analysis', 'General', 'scatter_Lfit_Pgen', 'max',
                          painting + '.png')
    plt.savefig(fig_fp)
    plt.close()
    return fig


def multiple_scatterfit_GEN_plot(V_p, MSE_vals, V_tots):
    # Reorganize data for plotting
    offset = N_paintings - 1
    temp = [[MSE_vals[i + (offset * j)] for i in range(len(V_tots))] for j in range(N_paintings)]
    for i in range(len(temp)):
        MSE_val = np.array([[temp[i][j][k] for j in range(len(temp[i]))]
                            for k in range(4)])
        MSE_val = MSE_val.reshape(4, -1)
        MSE_val = MSE_val.reshape(4, -1)
        y_std = np.std(MSE_val, axis=1)
        y_mean = np.mean(MSE_val, axis=1)
        x = np.array([V_p[i] for i in range(len(MSE_val)) for j in range(len(MSE_val[i]))])
        y = MSE_val.flatten()

        # Create a regression line model of data
        X, Y = x.reshape(-1, 1), y.reshape(-1, 1)
        regr = LinearRegression()
        regr.fit(X, Y)
        X_uniq = np.unique(x).reshape(-1, 1)
        Y_pred = regr.predict(X_uniq)
        X_uniq, Y_pred = X_uniq.flatten(), Y_pred.flatten()

        # Plot the scatter and fitted line
        fig = plt.figure()
        scatter_plot(x, y, 'Data points')
        plt.plot(X_uniq, Y_pred, color='k', label='trend line')
        # plt.plot(X_uniq, y_mean, color='blue', label='mean')  # If including mean
        plt.fill_between(X_uniq, y_mean - y_std, y_mean + y_std, color='blue', alpha=0.3, label='standard deviation')
        title = 'General trend all paintings for ' + str(V_tots[i % len(V_tots)]) + ' Vertices'
        plt.legend(loc='upper left')
        plt.title(title)
        plt.xlabel('Vertices Per Polygon')
        plt.ylabel('% Offset of mean MSE')
        fig_fp = os.path.join('Analysis', 'General', 'scatter_Lfit_Vgen', 'max',
                              str(V_tots[i % len(V_tots)]) + '.png')
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
    MSE_scaled_all = []
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

            # all_max = max(sample_MSE_list.flatten())
            # MSE_scaled = sample_MSE_list / all_max

            # calculate mean and standard deviation
            MSE_std = np.std(MSE_scaled, axis=1)
            MSE_mean = np.mean(MSE_scaled, axis=1)

            if singular:
                singular_plot(V_p, MSE_mean, MSE_std, paintings[i], V_tot[j])
            if multiple_P or multiple_V:
                multiple_means.append(MSE_mean)
                multiple_std.append(MSE_std)

            if scatter_S:
                singular_fitted_scatter_plot(V_p, MSE_scaled, paintings[i], V_tot[j])
            else:
                MSE_scaled_all.append(MSE_scaled)

        # Plot multiple V_tot per painting
        if multiple_V:
            multiple_V_plot(V_p, multiple_means, multiple_std, paintings[i],
                            V_tot)
            multiple_means = []
            multiple_std = []

        if scatter_M_gen:
            multiple_scatterfit_gen_plot(V_p, MSE_scaled_all, paintings[i])
            MSE_scaled_all = []

    if scatter_M_Vgen:
        multiple_scatterfit_GEN_plot(V_p, MSE_scaled_all, V_tot)
        MSE_scaled_all = []

    if multiple_P:
        multiple_P_plot(V_p, multiple_means, multiple_std, paintings,
                        V_tot)
        multiple_means = []
        multiple_std = []





        # Plot and store graph results single painting multiple V_tot
    # plot and store graph showing results all paintings, V_tot & V_p
