import heapq
import numpy as np

from agent import Agent
from game import Game
from evolution import Evolution


gen_pop = 20
parent_pop = 5
brain_shape = (2, 9, 9, 9, 2)
population = [Agent(brain_shape) for _ in range(gen_pop)]
evolve_iter = 20

evol = Evolution(population)

dummy_game = Game.rand_inst()
dummy_game.iter = 100

ranking = evol.eval_gen(dummy_game)
print(f'first: \n{np.array(ranking[:parent_pop])} \n\n\n')

evol.evolve(evolve_iter, parent_pop, 100)

ranking = evol.eval_gen(dummy_game)
print(f'last:\n{np.array(ranking[:parent_pop])}')

#print('\n', len(Evolution.new_gen(np.array(ranking[:parent_pop])[:, 2], 1)))