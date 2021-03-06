import os
import multiprocessing as mp
import helpers as hlp


# Create path to all paintings in paintings folder
source_folder = 'paintings'
paintings = [os.path.join(source_folder, f) for f in os.listdir(source_folder)]

# Estimate adequate safepoints for later analysis of data
savepoints = list(range(0, 250000, 1000)) + list(range(250000, 1000000, 10000))
stepsize = 100
repetitions = 2
V_polygon = [3, 4, 5, 6]
V_total = [1500, 1800]
iterations = 1000000
res_folder = 'Results2'
worker_count = 15   # amount of cores available on one node on Lisa server
N_p = len(paintings) # amount of artworks to be approximated
names = [p.split('/')[1].split('-')[0] for p in paintings]

if __name__ == '__main__':
    logfile, datafile, total_runs = hlp.init_folder_structure("HC", paintings,
                                                              repetitions,
                                                              V_total,
                                                              V_polygon,
                                                              iterations,
                                                              main_res_folder=res_folder)
    for i in range(worker_count):
        i %= N_p
        args = (names[i], "HC", [paintings[i]], repetitions, V_total, iterations,
                savepoints, V_polygon, logfile, datafile, total_runs, stepsize,
                res_folder)
        p = mp.Process(target=hlp.experiment, args=args)
        p.start()
