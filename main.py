import time
import math
import os
import multiprocessing as mp
import csv
import helpers as hlp


def experiment(name, algorithm, paintings, repetitions, V_total, iterations,
               savepoints, V_polygon):
    # get date/time
    now = time.strftime("%c")

    # create experiment directory with log .txt file
    folder = os.path.join("Results", name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Update name to be algorithm specific
    name = algorithm

    # logging experiment metadata
    total_runs = len(V_total) * len(paintings) * repetitions
    logfile = os.path.join(folder, name + "-LOG.txt")
    hlp.log_test_statistics(logfile, name, now, iterations, paintings, V_total,
                            repetitions, total_runs)

    # initializing the main datafile
    datafile = os.path.join(folder, name + "-DATA.csv")
    if not os.path.exists(datafile):
        hlp.init_datafile(datafile)

    # main experiment, looping through repetitions, vertex numbers, and paintings:
    for painting in paintings:
        painting_name = painting.split("/")[1].split("-")[0]
        for V_tot in V_total:
            n = (painting_name + "-" + algorithm + "_" + str(V_polygon) + "_" +
                 str(V_tot))
            # existing =
            for repetition in range(1, repetitions + 1):
                start = time.time()
                # make a directory for this run, containing the per iteration data and a selection of images
                n = n + "_" + str(repetition)
                outdir = os.path.join(folder, n)
                os.makedirs(outdir)

                # run the solver with selected algorithm
                solver = hlp.solver_select(painting, algorithm, V_tot, V_polygon,
                                           savepoints, outdir, iterations,
                                           population_size, nmax)
                solver.run()
                solver.write_data()

                bestMSE = solver.best.fitness

                # save best value in maindata sheet
                datarow = [painting_name, str(V_tot), str(V_polygon), str(repetition), bestMSE]

                with open(datafile, 'a', newline = '') as f:
                    writer = csv.writer(f)
                    writer.writerow(datarow)

                end = time.time()
                now = time.strftime("%c")
                with open(logfile, 'a') as f:
                    f.write(now + " finished run " + str(repetition) + "/" +
                            str(total_runs) + " V_total: " + str(V_tot) +
                            " painting: " + painting_name + " in " +
                            str((end - start)/60) + " minutes\n")


# paintins = ["paintings/monalisa-240-180.png", "paintings/bach-240-180.png", "paintings/dali-240-180.png", "paintings/mondriaan2-180-240.png", "paintings/pollock-240-180.png", "paintings/starrynight-240-180.png"]
paintins = ["paintings/kiss-180-240.png"]
savepoints = list(range(0, 250000, 1000)) + list(range(250000, 1000000, 10000))
repetitions = 1
V_total = [60, 300, 600]
V_polygon = 3
# V_total = [60, 120, 180, 240, 300, 600]
iterations = 10000
# define a list of savepoints, more in the first part of the run, and less later.
# savepoints = list(range(0, 2500, 50)) + list(range(2500, 10000, 500))

population_size = 30
nmax = 5

names = [p.split('/')[1].split('-')[0] for p in paintins]

#experiment(name, "HC" paintins, repetitions, V_total, iterations, savepoints)

# parallelize stuff

if __name__ == '__main__':
    worker_count = 1
    worker_pool = []
    for i in range(worker_count):
        args = (names[i], "HC", paintins, repetitions, V_total, iterations, savepoints, V_polygon)
        p = mp.Process(target=experiment, args=args)
        p.start()
