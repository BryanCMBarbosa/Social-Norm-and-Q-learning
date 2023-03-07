#ifndef SOCIALAGENT_H
#define SOCIALAGENT_H

#include "Agent.h"
#include <iostream>
#include <bitset>
#include <vector>
#include <queue>
#include <numeric>
#include <random>

#define strategy_length 8

using namespace std;

class SocialAgent: public Agent
{
    public:
        SocialAgent(unsigned long id, vector<float> epsilon = {0.01, 0.99}, vector<float> chi = {0.01, 0.99});
        short repcomb_to_index(Agent& receptor);
        bool act(Agent &receptor);
        void generate_strategy();
        void reset();
        void add_payoff(double payoff);
        void add_reputation(bool rep);

        unsigned long id;
        bitset<strategy_length> strategy;
        double payoffs_sum;
        double fitness;
        vector<float> epsilon;
        vector<float> chi;

    private:
        mt19937 mt;
};

#endif