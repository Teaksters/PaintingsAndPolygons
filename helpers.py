import algorithms as alg

import csv
from PIL import Image
import numpy as np


def log_test_statistics(logfile, name, now,
                        iterations, paintings, V_tot,
                        repetitions, total_runs):
    '''Writes test statistics to a logfile for later reference.'''
    with open(logfile, 'a') as f:
        f.write("EXPERIMENT " + name + " LOG\n")
        f.write("DATE " + now + "\n\n")
        f.write("STOP CONDITION " + str(iterations) + " iterations\n\n")
        f.write("LIST OF PAINTINGS (" + str(len(paintings)) + ")\n")
        for painting in paintings:
            f.write(painting + "\n")
        f.write("\n")
        f.write("VERTICES " + str(len(V_tot)) + " " + str(V_tot) + "\n\n")
        f.write("REPETITIONS " + str(repetitions) + "\n\n")
        f.write("RESULTING IN A TOTAL OF " + str(total_runs) + " RUNS\n\n")
        f.write("STARTING EXPERIMENT NOW!\n")
    f.close()


def init_datafile(datafile):
    '''Initializes datafile meant for containing test data.'''
    header = ["Painting", "Vertices", "Vertices Per Polygon", "Replication",
              "MSE"]
    with open(datafile, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    f.close()


def solver_select(painting, algorithm, V_total, V_polygon,
                  savepoints, outdir, iterations,
                  population_size, nmax, EMM="MSE"):
    '''intializes a solver class according to the selected algorithm and
    error measurement method.'''
    # Set goal for algorithm
    im_goal = Image.open(painting)
    goal = np.array(im_goal)
    h, w = np.shape(goal)[0], np.shape(goal)[1]

    # create corresponding solver object
    if algorithm == "PPA":
        poly = V_total / V_polygon
        nparam = (poly * 4 * 2) + (poly * 4) + poly
        mmax = math.ceil(nparam * 0.10)

        solver = alg.PPA(goal, w, h, V_total / V_polygon, V_total, EMM, savepoints, outdir, iterations, population_size, nmax, mmax)

    elif algorithm == "HC":
        solver = alg.Hillclimber(goal, w, h, V_total / V_polygon, V_total, EMM, savepoints, outdir, iterations)

    elif algorithm =="SA":
        solver = alg.SA(goal, w, h, V_total / V_polygon, V_total, EMM, savepoints, outdir, iterations)
    return solver
