
# Import external libraries
import csv
from PIL import Image
import numpy as np
import os
import time
import math

# Import custom libraries
import algorithms as alg


def log_test_statistics(logfile, name, now, iterations, paintings, V_tot,
                        repetitions, total_runs, V_p):
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
        f.write("V_PER_POLYGON " + str(len(V_p)) + " " + str(V_p) + "\n\n")
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
                  population_size, nmax, stepsize=0, EMM="MSE"):
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

        solver = alg.PPA(goal, w, h, V_total / V_polygon, V_total, EMM,
                         savepoints, outdir, iterations, population_size, nmax,
                         mmax)

    elif algorithm == "HC":
        solver = alg.Hillclimber(goal, w, h, V_total / V_polygon, V_total,
                                 EMM, savepoints, outdir, iterations,
                                 stepsize)

    elif algorithm == "SA":
        solver = alg.SA(goal, w, h, V_total / V_polygon, V_total, EMM,
                        savepoints, outdir, iterations)
    return solver


def init_folder_structure(algorithm, paintings, repetitions, V_total,
                          V_polygon, iterations, main_res_folder="Results"):
    '''Initializes all logging files and folders seperately from experiment
    allowing for experiment to be parallellized without printing headers again.
    '''
    # get date/time
    now = time.strftime("%c")

    # create experiment directory with log .txt file
    if not os.path.exists(main_res_folder):
        os.makedirs(main_res_folder)

    # Update name to be algorithm specific
    name = algorithm

    # logging experiment metadata
    total_runs = (len(V_total) * len(paintings) * repetitions * len(V_polygon))
    logfile = os.path.join(main_res_folder, name + "-LOG.txt")
    log_test_statistics(logfile, name, now, iterations, paintings, V_total,
                        repetitions, total_runs, V_polygon)

    # initializing the main datafile
    datafile = os.path.join(main_res_folder, name + "-DATA.csv")
    if not os.path.exists(datafile):
        init_datafile(datafile)

    # Return handles to files for in run data gathering
    return logfile, datafile, total_runs


def experiment(name, algorithm, paintings, repetitions, V_total, iterations,
               savepoints, V_polygon, logfile, datafile, total_runs,
               stepsize=0, main_res_folder="Results", worker=1,
               population_size=30, nmax=5):

    # main experiment, looping through repetitions, vertex numbers, and paintings:
    for painting in paintings:

        # create experiment directory for results and safepoints
        painting_name = painting.split("/")[1].split("-")[0]
        folder = os.path.join(main_res_folder, painting_name)
        if not os.path.exists(folder):
            os.makedirs(folder)

        for V_tot in V_total:
            for V_pol in V_polygon:
                n = (painting_name + "-" + algorithm + "_" + str(V_pol) + "_" +
                     str(V_tot))
                # existing =
                for repetition in range(1, repetitions + 1):
                    start = time.time()
                    # make a directory to contain iteration data + images
                    n = n + "_" + str(repetition)
                    outdir = os.path.join(folder, n)
                    # Update outdir to evade current existing results
                    while os.path.exists(outdir):
                        temp = outdir.split('_')
                        temp[-1] = str(int(temp[-1]) + 1)
                        outdir = '_'.join(temp)
                    os.makedirs(outdir)
                    # Make Checkpoints directory if storing MSE-checkpoints
                    if stepsize > 0:
                        os.makedirs(os.path.join(outdir, 'MSE_checkpoints'))

                    # run the solver with selected algorithm
                    solver = solver_select(painting, algorithm, V_tot,
                                           V_pol, savepoints, outdir,
                                           iterations, population_size, nmax,
                                           stepsize)
                    solver.run()
                    solver.write_data()

                    # save best value in main data sheet
                    bestMSE = solver.best.fitness
                    datarow = [painting_name, str(V_tot), str(V_pol),
                               str(repetition), bestMSE]

                    with open(datafile, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(datarow)

                    end = time.time()
                    now = time.strftime("%c")

                    # logging experiment metadata
                    with open(logfile, 'r') as f:
                        lines = f.read().splitlines()
                        last_line = lines[-1]
                        if len(last_line.split()) < 5:
                            current_run = 1
                        else:
                            current_run = int(last_line.split()[7].split('/')[0]) + 1
                    f.close()

                    with open(logfile, 'a') as f:
                        f.write(str(now) + " finished run " + str(current_run)
                                + "/" + str(total_runs) + " V_total: " +
                                str(V_tot) + " painting: " + painting_name +
                                " in " + str(round((end - start)/60, 2)) +
                                " minutes.\n")
                    f.close()
