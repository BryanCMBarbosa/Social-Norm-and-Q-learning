#ifndef QAGENT_H
#define QAGENT_H

#include "Agent.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

class QAgent: public Agent
{
    public:
        QAgent(long id, int num_states = 16, std::vector<bool> actions = {true, false}, double learning_rate = 0.2, double discount_factor = 0.95, double temperature = 5.0, double exploration_annealing_factor = 0.999, int initial_state = 15);
        bool act(Agent& receptor);
        void add_payoff(double payoff);
        void add_reputation(bool rep);
        void set_new_state(int new_state);
        void change_state();
        void reset();

        std::vector<int> states;
        std::vector<bool> actions;
        std::vector<std::vector<double>> q_table;
        int num_states;
        int num_actions;
        double learning_rate;
        double discount_factor;
        long current_iteration;
        double temperature;
        bool receptor_current_rep;
        bool receptor_previous_rep;
        int exploitation_acts;
        
    private:
        void config_q_table();
        void print_q_table();
        
        std::mt19937 mt;
        double initial_temperature;
        double exploration_annealing_factor;
        int initial_state;
        int previous_state;
        int current_state;
        int next_state;
        int arg_action_taken;
        std::vector<double> payoffs;
};

#endif