from random import randint, choice, shuffle
from PIL import Image
import numpy as np
from algorithms import Algorithm, Hillclimber, SA, PPA
import time
import math
import os
import csv
from multiprocessing import Process, current_process


# im_goal = Image.open("paintings/bach-240-180.png")
# im_goal = Image.open("paintings/dali-240-180.png")
im_goal = Image.open("paintings/monalisa-240-180.png")
# im_goal = Image.open("paintings/pollock-240-180.png")
# im_goal = Image.open("paintings/mondriaan2-180-240.png")


goal = np.array(im_goal)
h, w = np.shape(goal)[0], np.shape(goal)[1]
method = "MSE"
# outdirx = "test/"


# ppa specific settings
population_size = 30
nmax = 5  # max number of runners for the best indidiviual within a population


def experiment(name, algorithm, paintings, repetitions, polys, iterations, savepoints, V_p):
    # get date/time
    now = time.strftime("%c")

    # create experiment directory with log .txt file
    folder = os.path.join("Results", name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    total_runs = len(polys) * len(paintings) * repetitions

    # logging a lot of metadata
    logfile = os.path.join(folder, name + "-LOG.txt")

    with open(logfile, 'a') as f:
        f.write("EXPERIMENT " + name + " LOG\n")
        f.write("DATE " + now + "\n\n")
        f.write("STOP CONDITION " +str(iterations) + " iterations\n\n")
        f.write("LIST OF PAINTINGS (" + str(len(paintings)) +")\n")
        for painting in paintings:
            f.write(painting + "\n")
        f.write("\n")
        f.write("POLYS " + str(len(polys)) + " " + str(polys) + "\n\n")
        f.write("REPETITIONS " +str(repetitions) + "\n\n")
        f.write("RESULTING IN A TOTAL OF " + str(total_runs) + " RUNS\n\n")
        f.write("STARTING EXPERIMENT NOW!\n")

    # initializing the main datafile
    datafile = os.path.join(folder, name + "-DATA.csv")
    header = ["Painting", "Vertices", " Replication", "MSE"]
    with open(datafile, 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(header)


    # main experiment, looping through repetitions, poly numbers, and paintings:
    exp = 1
    for painting in paintings:
        painting_name = painting.split("/")[1].split("-")[0]
        for poly in polys:
            for repetition in range(repetitions):
                tic = time.time()
                # make a directory for this run, containing the per iteration data and a selection of images
                outdir = os.path.join(folder, str(exp) + "-" + str(repetition) + "-" + str(poly) + "-" + painting_name)
                os.makedirs(outdir)

                # run the hillclimber
                im_goal = Image.open(painting)
                goal = np.array(im_goal)
                h, w = np.shape(goal)[0], np.shape(goal)[1]

                if algorithm == "PPA":
                    nparam = (poly * 4 * 2) + (poly * 4) + poly
                    mmax = math.ceil(nparam * 0.10)

                    solver = PPA(goal, w, h, poly, poly * V_p, "MSE", savepoints, outdir, iterations, population_size, nmax, mmax)

                elif algorithm == "HC":
                    solver = Hillclimber(goal, w, h, poly, poly * V_p, "MSE", savepoints, outdir, iterations)

                elif algorithm =="SA":
                    solver = SA(goal, w, h, poly, poly * V_p, "MSE", savepoints, outdir, iterations)

                # run the solver with selected algorithm
                solver.run()
                solver.write_data()

                for poly in solver.best.genome:
                    print(len(poly[0]))

                bestMSE = solver.best.fitness

                # save best value in maindata sheet
                datarow = [painting_name, str(poly * V_p), str(repetition), bestMSE]

                with open(datafile, 'a', newline = '') as f:
                    writer = csv.writer(f)
                    writer.writerow(datarow)

                toc = time.time()
                now = time.strftime("%c")
                with open(logfile, 'a') as f:
                    f.write(now + " finished run " + str(exp) + "/" + str(total_runs) + " n: " + str(repetition) + " poly: " + str(poly) + " painting: " + painting_name + " in " + str((toc - tic)/60) + " minutes\n")

                exp += 1



name = "1miltest.x2"
# paintins = ["paintings/monalisa-240-180.png", "paintings/bach-240-180.png", "paintings/dali-240-180.png", "paintings/mondriaan2-180-240.png", "paintings/pollock-240-180.png", "paintings/starrynight-240-180.png"]
paintins = ["paintings/kiss-180-240.png"]
savepoints = list(range(0, 250000, 1000)) + list(range(250000, 1000000, 10000))
repetitions = 1
polys = [60]
V_p = 6
# polys = [60, 120, 180, 240, 300, 600]
iterations = 10000
# define a list of savepoints, more in the first part of the run, and less later.
# savepoints = list(range(0, 2500, 50)) + list(range(2500, 10000, 500))

population_size = 30
nmax = 5


args = (name, paintins, repetitions, polys, iterations, savepoints)

names = ["kiss1.6", "kiss2", "kiss3", "kiss4", "kiss5", "kiss6"]

#experiment(name, "HC" paintins, repetitions, polys, iterations, savepoints)

# parallelize stuff

if __name__ == '__main__':
    worker_count = 1
    worker_pool = []
    for i in range(worker_count):
        args = (names[i], "HC", paintins, repetitions, polys, iterations, savepoints, V_p)
        p = Process(target=experiment, args=args)
        p.start()
