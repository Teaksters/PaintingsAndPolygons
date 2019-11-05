'''
Code for heuristic solver algorithms can be found here, including a
supporting datastructure, the Algorithm class. Which handles the data and its
storage. This class is used by the heuristic classes to create solutions.
The following heuristic solvers are implemented:

Hillclimber - Sometimes also referred to as Local Search algorithm.
Simulated Annealing (SA) - Hillclimber with chance to accept deteriorations in
scores depending on a temperature variable.
Plan Propagation Algorithm (PPA) - A genetic algorithm managing its population
based on plant propagation models.
'''


# Import external libraries
import copy
import numpy as np
import math
import random
import csv
import os

# Import custom libraries
import constellation as c
import population as p


class Algorithm():
    '''Datastructure holding the data for the goal image and is responsible for
    storing the data received during the running of heuristic methods.'''
    def __init__(self, goal, w, h, num_poly, num_vertex, comparison_method,
                 savepoints, outdirectory, stepsize):
        self.goal = goal
        self.goalpx = np.array(goal)
        self.w = w
        self.h = h
        self.num_poly = num_poly
        self.num_vertex = num_vertex
        self.comparison_method = comparison_method
        self.data = []
        self.savepoints = savepoints
        self.outdirectory = outdirectory
        self.checks_out = os.path.join(self.outdirectory, 'Checkpoints')
        self.check_point = np.nan
        self.stepsize = stepsize

    def save_data(self, row):
        '''function to be called every safepoint to store running data of that
        iteration'''
        self.data.append(row)

    def write_data(self):
        '''writes all collected running data to a file'''
        file = os.path.join(self.outdirectory, 'data.csv')
        with open(file, 'a', newline='') as f:
            writer = csv.writer(f)
            for row in self.data:
                writer.writerow(row)


class Hillclimber(Algorithm):
    '''Hillclimber class used for running a hillclimbing solver algorithm capped
    by iterations. Uses the Constellation class to store states and do
    adaptations to those states moving closer to the optimal solution.
    Additionally it uses the Algorithm class to record the running data.'''
    def __init__(self, goal, w, h, num_poly, num_vertex, comparison_method,
                 savepoints, outdirectory, iterations, stepsize=100):
        # Initialize Algorithm class
        super().__init__(goal, w, h, num_poly, num_vertex, comparison_method,
                         savepoints, outdirectory, stepsize)
        self.iterations = iterations

        # initializing solution datastructure
        self.best = c.Constellation(0, 0, None, self.w, self.h)
        self.best.initialize_random_vertices(int(num_vertex / num_poly),
                                             num_vertex)
        self.best.img_to_array()
        self.best.calculate_fitness_mse(self.goalpx)
        self.check_point = self.best.fitness

        # define data header for hillclimber
        self.data.append(["Polygons", "Iteration", "MSE"])

    def run(self):
        for i in range(self.iterations):
            if i % 100 == 0:
                print(i)
            state = c.Constellation(0, 0, None, self.w, self.h)
            state.polygons = self.best.deepish_copy_state()
            state.id = i
            state.random_mutation(1)
            state.img_to_array()

            if self.comparison_method == "MSE":
                state.calculate_fitness_mse(self.goalpx)

            if state.fitness <= self.best.fitness:
                self.best = copy.deepcopy(state)

            # Store data per iteration
            if i in self.savepoints:
                self.best.save_img(self.outdirectory)
                self.best.save_polygons(self.outdirectory)
                self.save_data([int(self.num_poly), i,
                                round(self.best.fitness, 2)])

            # Store data per MSE improvement (for better movies etc.)
            if self.best.fitness <= self.check_point:
                self.best.save_img(self.checks_out)
                self.best.save_polygons(self.checks_out)
                while self.best.fitness <= self.check_point:
                    self.check_point -= self.stepsize


        self.best.save_img(self.outdirectory)
        self.best.save_polygons(self.outdirectory)
        self.save_data([int(self.num_poly), i, round(self.best.fitness, 2)])

class SA(Algorithm):
    '''Simulated Annealing class used for running a hillclimbing solver
    algorithm capped by iterations. Uses the Constellation class to store
    states and do adaptations to those states moving closer to the optimal
    solution. Additionally it uses the Algorithm class to record the running
    data.

    source: https://am207.github.io/2017/wiki/lab4.html'''
    def __init__(self, goal, w, h, num_poly, num_vertex, comparison_method,
                 savepoints, outdirectory, iterations):
        super().__init__(goal, w, h, num_poly, num_vertex, comparison_method,
                         savepoints, outdirectory)
        self.iterations = iterations

        # initializing organism
        self.best = c.Constellation(0, 0, None, self.w, self.h)
        self.best.initialize_genome(self.num_poly, num_vertex)
        self.best.img_to_array()
        self.best.calculate_fitness_mse(self.goalpx)

        self.current = copy.deepcopy(self.best)

        # define data header for SA
        self.data.append(["Polygons", "Iteration", "bestMSE", "currentMSE"])

    def acceptance_probability(self, dE, T):
        '''Returns the acceptance probability given T (Temperature) and
        dE (difference in score.)'''
        return math.exp(-dE/T)

    def cooling_geman(self, i):
        '''Returns temperature given current iteration.
        Remaining question: What to pick for C?
        d is usually set to one according to Nourani & Andresen (1998)'''
        c = 195075
        d = 1

        return c/math.log(i + d)

    def run(self):
        for i in range(1, self.iterations):
            state = c.Constellation(0, i, None, self.w, self.h)
            state.polygons = self.current.deepish_copy_state()
            state.random_mutation(1)
            state.img_to_array()

            state.calculate_fitness_mse(self.goalpx)


            dE = state.fitness - self.current.fitness
            T = self.cooling_geman(i)

            acceptance = self.acceptance_probability(dE, T)

            if random() < acceptance:
                self.current.genome = state.deepish_copy_state()
                self.current.fitness = state.fitness

            if self.current.fitness < self.best.fitness:
                self.best = copy.deepcopy(self.current)

            if i in self.savepoints:
                self.best.generation = i
                self.best.save_img(self.outdirectory)

            self.save_data([self.num_poly, i, self.best.fitness, self.current.fitness])

        # self.best.save_img(self.outdirectory)


class PPA(Algorithm):
    def __init__(self, goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory, iterations, pop_size, nmax, mmax):
        super().__init__(goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory)
        self.iterations = iterations
        self.pop_size = pop_size
        self.nmax = nmax
        self.mmax = mmax
        self.evaluations = 0
        self.pop = p.Population(self.pop_size)
        self.best = None
        self.worst = None

        # define data header for hillclimber
        self.data.append(["Polygons", "Generation", "Evaluations", "bestMSE", "worstMSE", "medianMSE", "meanMSE"])

        # fill population with random polygon drawings
        for i in range(self.pop_size):
            alex = c.Constellation(0, i, None, self.w, self.h)
            alex.initialize_genome(self.num_poly, self.num_vertex)
            alex.img_to_array()
            alex.calculate_fitness_mse(self.goalpx)
            self.pop.add_organism(alex)

    def calculate_random_runners(self):
        # if the populations max and min fitness are equal, this function generates random runners and distance for all organisms
        for organism in self.pop.organisms:
            organism.nr = random.randint(1, self.nmax)
            organism.d = random.randint(1, self.mmax)

    def calculate_runners(self):
        # default runner calculation for all organisms in the population
        for organism in self.pop.organisms:
            organism.scale_fitness(self.worst.fitness, self.best.fitness)
            organism.calculate_runners(self.nmax, self.mmax)

    def generation(self, gen):
        counter = 0
        for organism in self.pop.organisms[:]:
            for i in range(organism.nr):
                state = c.Constellation(gen, counter, organism.name(), self.w, self.h)
                state.polygons = organism.deepish_copy_state()
                state.random_mutation(organism.d)
                state.img_to_array()
                state.calculate_fitness_mse(self.goalpx)

                self.pop.add_organism(state)
                counter += 1

        self.evaluations += counter


    def run(self):
        gen = 1

        while self.evaluations < self.iterations:
            self.pop.sort_by_fitness()
            self.best = self.pop.return_best()
            self.worst = self.pop.return_worst()

            if gen in self.savepoints:
                self.best.save_img(self.outdirectory)

            if self.best.fitness != self.worst.fitness:
                self.calculate_runners()
                self.generation(gen)
            else:
                self.calculate_random_runners()
                self.generation(gen)

            gen += 1


            self.pop.eliminate()
            best, worst, median, mean = self.pop.return_data()
            self.save_data([self.num_poly, gen, self.evaluations, best.fitness, worst.fitness, median, mean])


        self.pop.sort_by_fitness()
        self.best = self.pop.return_best()
        self.best.save_img(self.outdirectory)
