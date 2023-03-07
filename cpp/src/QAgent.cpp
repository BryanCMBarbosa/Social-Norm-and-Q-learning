#include "QAgent.h"

QAgent::QAgent(long id, int num_states, std::vector<bool> actions, double learning_rate, double discount_factor, double temperature, double exploration_annealing_factor, int initial_state) : mt((std::random_device())())
{
    this->id = id;
    this->num_states = num_states;
    this->num_actions = actions.size();
    this->actions = actions;
    config_q_table();
    this->initial_state = initial_state;
    this->current_state = initial_state;
    this->payoffs_sum = 0;
    this->fitness = 0;
    this->learning_rate = learning_rate;
    this->discount_factor = discount_factor;
    this->initial_temperature = temperature;
    this->temperature = temperature;
    this->exploration_annealing_factor = exploration_annealing_factor;
    this->current_iteration = 0;
    this->acts_on_episode = 0;
    this->exploitation_acts = 0;
    generate_reputation();
}

void QAgent::config_q_table()
{
    std::vector<double> q_row(num_actions, 0.0);
    for (int i = 0; i < num_states; i++)
        q_table.push_back(q_row);
}

bool QAgent::act(Agent& receptor)
{
    current_iteration++;
    temperature = initial_temperature * pow(exploration_annealing_factor, current_iteration);
    
    receptor_current_rep = receptor.reputation.back();
    receptor_previous_rep = receptor.reputation.front();

    double max;
    int max_id;

    acts_on_episode++;

    if (temperature < 0.01)
    {
        exploitation_acts++;
        for(int i = 0; i < num_actions; i++)
        {
            if (i == 0)
            {
                max = q_table[current_state][i];
                max_id = i;
            }
            if (q_table[current_state][i] > max)
            {
                max = q_table[current_state][i];
                max_id = i;
            }
        }
        arg_action_taken = max_id;
    }
    else
    {
        double temp_inv = 1.0/temperature;
        double sum;
        std::vector<double> boltzmann_table;
        boltzmann_table.push_back(exp(q_table[current_state][0]*temp_inv));
        boltzmann_table.push_back(exp(q_table[current_state][1]*temp_inv));
        sum = boltzmann_table[0] + boltzmann_table[1];
        boltzmann_table[0] /= sum;
        boltzmann_table[1] /= sum;
        std::discrete_distribution<> dist(boltzmann_table.begin(), boltzmann_table.end());
        arg_action_taken = dist(mt);
    }

    return actions[arg_action_taken];
}

void QAgent::set_new_state(int new_state)
{
    next_state = new_state;
}

void QAgent::add_payoff(double payoff)
{
    payoffs.push_back(payoff);

    if(acts_on_episode == 2)
    {
        set_new_state((8 * receptor_previous_rep) + 
            (4 * reputation.front()) + 
            (2 * receptor_current_rep) + 
            (1 * actions[arg_action_taken]));
        
        double total_payoff = std::accumulate(payoffs.begin(), payoffs.end(), 0.0);
        double max_value = *std::max_element(q_table[next_state].begin(), q_table[next_state].end());
        q_table[current_state][arg_action_taken] = (1 - learning_rate)*q_table[current_state][arg_action_taken] + learning_rate*(total_payoff + discount_factor*max_value);
        payoffs.clear();
        acts_on_episode = 0;
    }
}

void QAgent::add_reputation(bool rep)
{
    reputation.pop();
    reputation.push(rep);
}

void QAgent::change_state()
{
    previous_state = current_state;
    current_state = next_state;
}

void QAgent::reset()
{
    for (int i = 0; i < num_states; i++)
        for (int j = 0; j < num_actions; j++)
            q_table[i][j] = 0.0;

    temperature = initial_temperature;

    while (!reputation.empty())
        reputation.pop();

    for(int i = 0; i < 2; i++)
        reputation.push(true);

    current_state = initial_state;
    current_iteration = 0;
    exploitation_acts = 0;
}