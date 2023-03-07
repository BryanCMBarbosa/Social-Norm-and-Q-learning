import numpy as np
from cmath import nan
from Agent import Agent
import matplotlib.pyplot as plt

class AgentQ(Agent):
    def __init__(self, states, actions, initial_state:int):
        Agent.__init__(self, states, actions, initial_state)
        self.q_table = np.zeros((len(states), len(actions)))
        self.boltzmann_table = np.zeros((len(states), len(actions)))

    def calculate_boltzmann_matrix(self, temperature):
        if ((self.q_table > 7).any()):
            q_table_exp = self.q_table/temperature #np.exp
        else:
            q_table_exp = np.exp(self.q_table/temperature)

        self.boltzmann_table = q_table_exp/np.sum(q_table_exp, axis=1)[:, None]

    def play(self, temperature):
        if temperature >= 0.01:
            self.calculate_boltzmann_matrix(temperature)
            self.action = np.random.choice(self.actions, p = self.boltzmann_table[self.state, :])
        else:
            self.action = np.argmax(self.q_table[self.state, :])
        
        self.chosen_actions = np.append(self.chosen_actions, self.action)
        self.increment_coop_and_defect()

        #return self.action

    def decide_next_state(self, oponent_action):  #00 - CC - 0 / 01 - CB - 1 / 10 - BC - 2 / 11 - BB - 3 
        if self.action == oponent_action:
            if self.action == 0:
                self.next_state = 0
            else:
                self.next_state = 3
        else:
            if self.action == 0:
                self.next_state = 1
            else:
                self.next_state = 2

    def upgrade_score(self, learning_rate, discount_factor):
        self.q_table[self.state, self.action] = (1 - learning_rate)*self.q_table[self.state, self.action] + learning_rate*(self.rewards[-1] + discount_factor*np.max(self.q_table[self.next_state, :]))

    def upgrade_state(self):
        self.state = self.next_state

    def print_data(self):
        print("\nQ AGENT STATISTICS")
        print("Coops: ", self.coops)
        print("Defections: ", self.defects)
        print("Average points per game: ", np.average(self.rewards))
        print("Learned strategy for CC: ", np.argmax(self.q_table[0, :]))
        print("Learned strategy for CB: ", np.argmax(self.q_table[1, :]))
        print("Learned strategy for BC: ", np.argmax(self.q_table[2, :]))
        print("Learned strategy for BB: ", np.argmax(self.q_table[3, :]))
        print(self.q_table)
    
    def plot_data(self):
        plt.plot(self.chosen_actions, linewidth=0.8)
        plt.xlabel("Iteration")
        plt.ylabel("0: Cooperation | 1: Defection")
        plt.show()

    def reset(self):
        self.state = 0
        self.action = nan
        self.coops = 0
        self.defects = 0
        self.chosen_actions = np.array([])
        self.rewards = np.array([])
        self.next_state = nan
        self.q_table = np.zeros((len(self.states), len(self.actions)))
        self.boltzmann_table = self.boltzmann_table = np.array((len(self.states), len(self.actions)))

