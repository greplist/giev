#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitMutation

from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

matplotlib.use('tkagg')

# lab: 7 variant
start, end, eps = 0, 10, 0.001

indv_template = BinaryIndividual(ranges=[(start, end)], eps=eps)
population = Population(indv_template=indv_template, size=10).init()

selection = TournamentSelection()
crossover = UniformCrossover(pc=1.0, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[])


def npfunc():
    t = np.arange(start, end, eps)
    return t, np.sin(0.7 * np.pi * t - 1.4) * (1.7 * t + 1.5)


@engine.fitness_register
def fitness(indv):
    t, = indv.solution
    return math.sin(0.7 * math.pi * t - 1.4) * (1.7 * t + 1.5)


@engine.analysis_register
class ConsoleGraphAnalysis(OnTheFlyAnalysis):
    interval = 1
    master_only = True

    def setup(self, ng, engine):
        self.logger.info('Start generate: {}'.format(ng))

        plt.plot(*npfunc())

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x, y = best_indv.solution[0], engine.ori_fmax
        self.logger.info('Generation: {}, x: {:.3f} best fitness: {:.3f}'.format(g, x, y))

        plt.scatter(x, y)

    def finalize(self, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x, y = best_indv.solution[0], engine.ori_fmax
        self.logger.info('Optimal solution: ({:.3f}, {:.3f})'.format(x, y))

        plt.scatter(x, y)
        plt.annotate('optimal', (x, y,))
        plt.show()


if '__main__' == __name__:
    engine.run(ng=100)

