'''
File containing functionality for storing individual states. Namely,
a set of polygons and corresponding colors that form the solution.
Additionally, functionality for performing mutations to those states and
to calculate fitness and error scores can be found here.
'''

# importing external libraries
import numpy as np
from math import ceil
from PIL import Image, ImageDraw
from skimage.measure import compare_ssim as ssim
from random import randint, choice, random
from numba import njit
import os


class Constellation():
    '''Data structure class that holds a solution state and functionality to
    mutate or calculate its scores.'''
    def __init__(self, generation, c, parent, w, h):
        self.generation = generation
        self.id = c
        self.parent = parent
        self.w = w
        self.h = h
        self.polygons = []
        self.array = []
        self.fitness = float('inf')
        self.scaled_fitness = 0
        self.Nx = 0
        self.nr = 0
        self.d = 0

    def initialize_random_vertices(self, V_p, V_tot):
        '''Does the same as initialize_genome only ensures all polygons have V_p
        vertici. Initializes random start!

        params:
        V_p: int, Vertices per Polygon.
        V_tot: int, total amount of vertices used.

        !!! Assumes V_tot is a multiple of V_p'''
        N_p = int(V_tot / V_p)  # Compute number of polygons
        for i in range(0, N_p):
            poly = []
            # Assign vertices to polygon
            for i in range(V_p):
                xy = (randint(0, self.w), randint(0,self.h))
                poly.append(xy)
            # Assign random RGBA-values
            color = (randint(0, 255),
                     randint(0, 255),
                     randint(0, 255),
                     randint(0, 255))
            # Add polygon data to final combination
            polygon = (poly, color)
            self.polygons.append(polygon)

    def calculate_fitness_mse(self, goal):
        '''Calculates the mse between current state and goal.'''
        self.fitness = mse(self.array, goal)

    def calculate_fitness_ssim(self, goal):
        '''Calculates the structural similarity index image difference between
        current state and goal. (higher = more similar)'''
        ssim_index = ssim(self.array, goal, multichannel=True)
        self.fitness = ssim_index

    def scale_fitness(self, minimum, maximum):
        '''Scales the fitness to be represented between 0 and 1.'''
        self.scaled_fitness = (self.fitness - minimum) / (maximum - minimum)

    def calculate_runners(self, nmax, mmax):
        # see Salhi & Fraga 2011

        # mapping fitness to emphasize good solutions
        self.Nx = 0.5 * (np.tanh(4 * self.scaled_fitness - 2) + 1)

        # calculate number of runners (nr) and the distance in number of mutations (d)
        r = random()
        self.nr = int(ceil(nmax * self.Nx * r))
        self.d = int(ceil(mmax * (1 - self.Nx) * r))

    ######################################################
    # Mutations
    ######################################################

    def random_mutation(self, number):
        '''Performs a number of random mutations on the current stateself.

        params:
        - number: int, number of mutations to perform.'''
        options = [self.draw_order_change,
                   self.move_vertex,
                   self.change_color  # ,

                   # this options is commented as it changes the vertices a polygon has.
                   # self.transfer_vertex
                   ]

        for i in range(0, number):
            mutation = choice(options)
            mutation()

    def draw_order_change(self):
        '''Changing the drawing order of a randomly selected polygon.'''
        i = randint(0, len(self.polygons) - 1)
        j = randint(0, len(self.polygons) - 1)

        polygon = self.polygons[i]
        del self.polygons[i]
        self.polygons.insert(j, polygon)

    def move_vertex(self):
        '''Change the location of a randomly selected vertex.'''
        xy = (randint(0,self.w), randint(0,self.h))
        i = randint(0, len(self.polygons) - 1)
        v = randint(0, len(self.polygons[i][0]) - 1)
        self.polygons[i][0][v] = xy

    def transfer_vertex(self):
        '''Transfer a randomly selected vertex from one polygon to a random
        other one.'''
        giver = 0
        receiver = 0

        # ensure different indexes and ensure that the giver has > 3 vertices
        while True:
            giver = randint(0, len(self.polygons) - 1)
            receiver = randint(0, len(self.polygons) - 1)
            if giver != receiver and len(self.polygons[giver][0]) > 3:
                break

        # pick a vertex from the giver and delete it
        n = randint(0, len(self.polygons[giver][0]) - 1)
        del self.polygons[giver][0][n]

        # pick two neighbouring vertices from the receiver and interpolate a new (x,y) coordinate between them
        i = randint(0,len(self.polygons[receiver][0]) - 2)
        xy1 = self.polygons[receiver][0][i]
        xy2 = self.polygons[receiver][0][i + 1]

        # calculate the slope of the line between xy1 and xy2
        slope = (xy1[1] - xy2[1]) / (xy1[0] - xy2[0] + 0.00001)

        # pick a random x between x1 and x2, and calculate correponding y. round.
        if xy1[0] < xy2[0]:
            x = randint(xy1[0], xy2[0])
            dx = x - xy1[0]
            y = int(round(dx * slope)) + xy1[1]
        else:
            x = randint(xy2[0], xy1[0])
            dx = x - xy2[0]
            y = int(round(dx * slope)) + xy2[1]

        xy_new = (x, y)
        self.polygons[receiver][0].insert(i + 1, xy_new)

    def change_color(self):
        '''changes the color (or the alpha) of a random polygon.'''
        i = randint(0, len(self.polygons) - 1)
        j = randint(0, 4)

        if j == 0:
            color = (randint(0, 255), self.polygons[i][1][1],
                     self.polygons[i][1][2], self.polygons[i][1][3])
        elif j == 1:
            color = (self.polygons[i][1][0], randint(0, 255),
                     self.polygons[i][1][2], self.polygons[i][1][3])
        elif j == 2:
            color = (self.polygons[i][1][0], self.polygons[i][1][1],
                     randint(0, 255), self.polygons[i][1][3])
        else:
            color = (self.polygons[i][1][0], self.polygons[i][1][1],
                     self.polygons[i][1][2], randint(0, 255))
        new_gene = (self.polygons[i][0], color)
        self.polygons[i] = new_gene

    ######################################################
    # Helpers
    ######################################################
    def save_img(self, directory):
        '''Store an image of the current state'''
        img_name = ("{:0>6}".format(self.generation) + "-" +
                    "{:0>3}".format(self.id) + "-" +
                    str(int(round(self.fitness, 0))) + ".png")
        filename = os.path.join(directory, img_name)
        im = Image.fromarray(self.array)
        im.save(filename)

    def save_polygons(self, directory):
        '''Store polygon data of the current state'''
        pol_name = ("{:0>6}".format(self.generation) + "-" +
                    "{:0>3}".format(self.id) + "-" +
                    str(int(round(self.fitness, 0))) + ".txt")
        filename = os.path.join(directory, pol_name)
        with open(filename, 'w') as f:
            for poly in self.polygons:
                    f.write(str(poly) + '\n')

    def name(self):
        return "{:0>6}".format(self.generation) + "-" + "{:0>3}".format(self.id)

    def img_to_array(self):
        '''Casts image to a numpy array.'''
        img = Image.new('RGB', (self.w, self.h), color=(0, 0, 0))
        drw = ImageDraw.Draw(img, 'RGBA')

        for polygon in self.polygons:
            color = polygon[1]
            vertices = polygon[0]
            drw.polygon(vertices, color)

        self.array = np.array(img)

    def deepish_copy_state(self):
        '''Create and return a copy of the current state.'''
        new_state = []

        for polygon in self.polygons:
            newpoly = []
            for vertex in polygon[0]:
                newpoly.append((vertex + tuple()))
            newcol = polygon[1] + tuple()
            new_poly = (newpoly, newcol)
            new_state.append(new_poly)

        return new_state


# mse function has to live outside of the class to be jitted
@njit
def mse(a, b):
    out = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                out += (a[i][j][k] - b[i][j][k]) ** 2

    out /= (a.shape[0]*a.shape[1])
    return out
