#ifndef AGENT_H
#define AGENT_H

#include <queue>
#include <iostream>

class Agent
{
    public:
        virtual bool act(Agent& receptor) = 0;
        virtual void add_payoff(double payoff) = 0;
        virtual void add_reputation(bool rep) = 0;
        virtual void reset() = 0;
        float get_fitness(long z);
        void generate_reputation();
        void reset_payoff();

        virtual ~Agent() = default;

        long id;
        double payoffs_sum;
        double fitness;
        std::queue<bool> reputation;
};

#endif