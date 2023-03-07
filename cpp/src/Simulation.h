#ifndef SIMULATION_H
#define SIMULATION_H

#include <iostream>
#include <vector>
#include <string>
#include <bitset>
#include <random>
//#include <omp.h>
#include <map>
#include <ctime> 
#include <math.h>
#include <fstream>
#include "Agent.h"
#include "QAgent.h"
#include "SocialAgent.h"

using namespace std;

#define strategy_length 8

class Simulation
{
    public:
        Simulation(bitset<16> norm, string norm_name, float payoff_b = 5, float payoff_c = 1, float alpha = 0.01);
        void run_generations(int runs);

    private:
        short reputation_combination_to_index(Agent& donor, Agent& receptor, bool donor_action);
        void judge(Agent& x, Agent& y, bool x_action, bool y_action);
        void donation_operation(Agent& x, Agent& y);
        void turn_to_csv(int runs, vector<vector<double>> eta_each_strategy);

        bitset<16> norm;
        string norm_name;
        unsigned long z;
        vector<Agent> agents;
        vector<float> mi;
        float payoff_b;
        float payoff_c;
        vector<float> alpha;
        vector<float> chi;
        vector<float> eta;
        unsigned long long coops;
        unsigned long long total_acts;
        int possible_strategies;
        bool keep_track;
        int available_threads;
        mt19937 mt;
};

#endif