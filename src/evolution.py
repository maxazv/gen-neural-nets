import heapq
import numpy as np

from agent import Agent
from game import Game

class Evolution:
    def __init__(self, pop) -> None:
        self.__pop = pop


    def eval_gen(self, game):
        ranking = []

        # threading here would be nice
        for agent in self.__pop:
            decision = agent.act(game.get_input(None))
            agent.set_score(game.play(decision))

            # have to add id(agent) if two agents have same score => 2nd item of tuple is compared
            heapq.heappush(ranking, [agent.score(), id(agent), agent])
        
        return ranking

    @staticmethod
    def new_gen(ranking, lr):
        new_gen = []

        # all possible pairs of the top ranking agents from last gen
        XX, YY = np.meshgrid(ranking, ranking)
        parent_pairs = np.vstack([XX.ravel(), YY.ravel()]).T

        # get new generation via crossover of the parent pairs
        for (p1, p2) in parent_pairs:
            child = Agent.crossover(p1, p2)
            child.mutate(lr)
            new_gen.append(child)

        return new_gen


    # this function doesnt really make sense in this class
    def evolve(self, it, parent_pop, game_it, func=np.sqrt):
        for i in range(it):
            game = Game.rand_inst()
            game.iter = game_it

            ranking = self.eval_gen(game)
            ranking = np.array(ranking[:parent_pop])

            self.__pop = Evolution.new_gen(ranking[:, 2], func(it-i+1))


    def pop(self):
        return self.__pop
    
    def set_pop(self, pop):
        self.__pop = pop
    
    def game(self):
        return self.__game