import heapq
import numpy as np

from agent import Agent
from game import Game


def play(velx, vely, game, iter=500):
    dist = float('inf')
    game.set_vel(np.array([velx, vely]))

    for i in range(iter):
        game.iteration()
        tmp = game.dist()
        if tmp < dist: dist = tmp

    game.reset()
    return dist


def eval_gen(pop, game, game_its):
    ranking = []

    # threading here would be nice
    for agent in pop:
        decision = agent.act(game.get_target())
        agent.set_score(play(decision[0][0], decision[1][0], game, iter=game_its))

        # have to add id(agent) if two agents have same score => 2nd item of tuple is compared
        heapq.heappush(ranking, [agent.score(), id(agent), agent])
    
    return ranking


def new_gen(ranking, rate=.3, freq=.2):
    new_gen = []

    # all possible pairs of the top ranking agents from last gen
    XX, YY = np.meshgrid(ranking, ranking)
    parent_pairs = np.vstack([XX.ravel(), YY.ravel()]).T

    # get new generation via crossover of the parent pairs
    for (p1, p2) in parent_pairs:
        child = Agent.crossover(p1, p2)
        child.mutate(rate, freq)
        new_gen.append(child)

    return new_gen


def evolve(population, game, parent_pop, it, game_its=500):
    for _ in range(it):
        ranking = eval_gen(population, game, game_its)
        ranking = np.array(ranking[:parent_pop])
        population = new_gen(ranking[:, 2])

    return population




gen_pop = 20
parent_pop = 5
brain_shape = (2, 9, 9, 2)
population = [Agent(brain_shape) for _ in range(gen_pop)]
dummy_game = Game(0, 200, res=1/50)
evolve_iter = 500

first_ranking = eval_gen(population, dummy_game, 100)
first_ranking = np.array(first_ranking[:parent_pop])

evolved_pop = evolve(population, dummy_game, parent_pop, evolve_iter, game_its=100)

last_ranking = eval_gen(evolved_pop, dummy_game, 100)
last_ranking = np.array(last_ranking)
print(f'First:\n {first_ranking} \n\n\nLast:\n {last_ranking}')