#include "Agent.h"

float Agent::get_fitness(long z)
{
    return payoffs_sum / (float)(2*z);
}

void Agent::generate_reputation()
{
    for(int i = 0; i < 2; i++)
        this->reputation.push(true);
}

void Agent::reset_payoff()
{
    fitness = 0.0;
    payoffs_sum = 0.0;
}