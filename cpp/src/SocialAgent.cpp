#include "SocialAgent.h"

SocialAgent::SocialAgent(unsigned long id, vector<float> epsilon, vector<float> chi) : mt((random_device())())
{
    this->id = id;
    this->strategy = bitset<strategy_length>(0);
    generate_reputation();
    fitness = 0.0;
    payoffs_sum = 0.0;
    this->epsilon = epsilon;
    this->chi = chi;
}

short SocialAgent::repcomb_to_index(Agent& receptor)
{   
    bernoulli_distribution dist(chi[0]);
    bitset<2> gossip_error;
    for(short i=0; i < gossip_error.size(); i++)
        gossip_error[i] = dist(mt);

    bool receptor_rep_1 = gossip_error[0] ? !receptor.reputation.back() : receptor.reputation.back();
    bool receptor_rep_2 = gossip_error[1] ? !receptor.reputation.front() : receptor.reputation.front();

    return (4 * receptor_rep_2) + (2 * this->reputation.back()) + (1 * receptor_rep_1);
}

bool SocialAgent::act(Agent &receptor)
{
    bernoulli_distribution dist(epsilon[0]);
    bool can_execute = dist(mt);

    if (can_execute)
        return strategy[repcomb_to_index(receptor)];
    else
        return !strategy[repcomb_to_index(receptor)];
}

void SocialAgent::reset()
{
    fitness = 0.0;
    payoffs_sum = 0.0;

    while (!reputation.empty())
        reputation.pop();
    
    for(int i = 0; i < 2; i++)
        reputation.push(true);
}

void SocialAgent::add_payoff(double payoff)
{
    payoffs_sum += payoff;
}

void SocialAgent::add_reputation(bool rep)
{
    reputation.pop();
    reputation.push(rep);
}