#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitBigMutation

from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

matplotlib.use('tkagg')

x1_range = (-2, 2)
x2_range = (-2, 2)
eps = 0.001

indv_template = BinaryIndividual(ranges=[x1_range, x2_range], eps=eps)
population = Population(indv_template=indv_template, size=10).init()

selection = TournamentSelection()
crossover = UniformCrossover(pc=1, pe=0.5)
mutation = FlipBitBigMutation(pm=0.1, pbm=0.55, alpha=0.6)

engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[])

ax = plt.axes(projection='3d')

def npfunc():
    x1 = np.linspace(*x1_range, num=100)
    x2 = np.linspace(*x2_range, num=100)
    X1, X2 = np.meshgrid(x1, x2)
    Y = X1 * X1 + 2 * X1 * X2 - 3 * np.cos(3 * np.pi * X1) * np.cos(4 * np.pi * X2) + 0.3
    return X1, X2, Y


def func(x1, x2):
    return (x1 * x1 + 2 * x1 * x2
                - 3 * math.cos(3 * math.pi * x1) * math.cos(4 * math.pi * x2) + 0.3)


@engine.fitness_register
def fitness(indv):
    x1, x2 = indv.solution
    return -func(x1, x2)


@engine.analysis_register
class ConsoleGraphAnalysis(OnTheFlyAnalysis):
    interval = 1
    master_only = True

    def setup(self, ng, engine):
        self.logger.info('Start generate: {}'.format(ng))

        ax.plot_surface(*npfunc(), rstride=1, cstride=1,
                        cmap='viridis', edgecolor='grey')

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x1, x2 = best_indv.solution
        y = func(x1, x2)
        self.logger.info('Generation: {}, x1: {:.3f} x2: {:.3f} y: {:.3f}'.format(
            g, x1, x2, y))

        ax.scatter(x1, x2, y, color='blue')

    def finalize(self, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x1, x2 = best_indv.solution
        y = func(x1, x2)
        self.logger.info('Optimal solution: ({:.3f}, {:.3f}, {:.3f})'.format(x1, x1, y))

        ax.scatter(x1, x2, y, color='red')
        ax.text(x1, x2, y + 0.1, 'optimal')
        plt.show()


if '__main__' == __name__:
    engine.run(ng=100)
