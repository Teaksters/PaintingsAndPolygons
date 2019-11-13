import os
import multiprocessing as mp
import helpers as hlp


# Create path to all paintings in paintings folder
source_folder = 'paintings'
paintings = [os.path.join(source_folder, f) for f in os.listdir(source_folder)]

# Estimate adequate safepoints for later analysis of data
savepoints = list(range(0, 250000, 1000)) + list(range(250000, 1000000, 10000))
stepsize = 100
repetitions = 1
# V_total = [60, 300, 600]
V_polygon = [3, 4, 5, 6]
# V_total = [60, 120, 180, 240, 300, 600, 1200]
V_total = [60, 120, 180]
iterations = 1000000

names = [p.split('/')[1].split('-')[0] for p in paintings]

if __name__ == '__main__':
    worker_count = len(paintings)
    logfile, datafile, total_runs = hlp.init_folder_structure("HC", paintings,
                                                              repetitions,
                                                              V_total,
                                                              V_polygon,
                                                              iterations)
    for i in range(worker_count):
        args = (names[i], "HC", [paintings[i]], repetitions, V_total, iterations,
                savepoints, V_polygon, logfile, datafile, total_runs, stepsize)
        p = mp.Process(target=hlp.experiment, args=args)
        p.start()
