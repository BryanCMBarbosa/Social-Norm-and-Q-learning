#include "Simulation.h"

Simulation::Simulation(bitset<16> norm, string norm_name, float payoff_b, float payoff_c, float alpha) : mt((random_device())())
{
    this->norm = norm;
    this->norm_name = norm_name;
    this->payoff_b = payoff_b;
    this->payoff_c = payoff_c;
    this->alpha = vector<float>{ alpha, 1-alpha };
    this->coops = 0;
    this->possible_strategies = pow(2, strategy_length);
    this->total_acts = 0;
    this->keep_track = false;
    //this->available_threads = omp_get_num_procs();
    //omp_set_num_threads(available_threads);
}

short Simulation::reputation_combination_to_index(Agent& donor, Agent& receptor, bool donor_action)
{
    return (8 * receptor.reputation.front()) + (4 * donor.reputation.front()) + (2 * receptor.reputation.back()) + (1 * donor_action);
}

void Simulation::judge(Agent& x, Agent& y, bool x_action, bool y_action)
{
    bernoulli_distribution dist(alpha[0]);
    bitset<2> can_assign;
    for(short i=0; i<can_assign.size(); i++)
        can_assign[i] = dist(mt);

    if (can_assign[0])
        x.add_reputation(norm[reputation_combination_to_index(x, y, x_action)]);
    else
        x.add_reputation(!norm[reputation_combination_to_index(x, y, x_action)]);

    if (can_assign[1])
        y.add_reputation(norm[reputation_combination_to_index(y, x, y_action)]);
    else
        y.add_reputation(!norm[reputation_combination_to_index(y, x, y_action)]);
}

void Simulation::donation_operation(Agent& x, Agent& y)
{
    bool x_act = x.act(y);
    bool y_act = y.act(x);
    total_acts += 2;

    if (x_act)
    {
        coops++;
        x.add_payoff(-payoff_c);
        y.add_payoff(payoff_b);
    }

    if (y_act)
    {
        coops++;
        y.add_payoff(-payoff_c);
        x.add_payoff(payoff_b);
    }

    judge(x, y, x_act, y_act);
}


void Simulation::run_generations(int runs)
{
    QAgent q(1);
    SocialAgent s(2);

    vector<vector<double>> eta_each_strategy;
    vector<double> eta_each_run;
    double eta;

    for(unsigned long long i = 0; i < possible_strategies; i++)
    {
        s.strategy = bitset<8>(i);
        cout << s.strategy << " started." << endl;

        for(int j = 0; j < runs; j++)
        {
            cout << endl << "Run " << j+1 << " started." << endl;

            while(q.temperature >= 0.01)
            {
                donation_operation(q, s);
                q.change_state();
            }

            eta = double(coops) / double(total_acts);
            eta_each_run.push_back(eta);
            coops = 0;
            total_acts = 0;

            q.reset();
            s.reset();

            cout << "Run " << j+1 << " finished." << endl;
            cout << "------------------------------------------" << endl;
        }
        eta_each_strategy.push_back(eta_each_run);
        eta_each_run.clear();
        cout << s.strategy << " finished." << endl;
        cout << "::::::::::::::::::::::::::::::::::::::::::" << endl;
    }
    cout << "Turning to CSV..." << endl;
    turn_to_csv(runs, eta_each_strategy);
    cout << "Done!" << endl;
}

void Simulation::turn_to_csv(int runs, vector<vector<double>> eta_each_strategy)
{
    fstream csv_file;
    string file_name = norm.to_string()+"_norm_name="+norm_name+"_runs="+to_string(runs)+".csv";
    csv_file.open(file_name, ios::out | ios::app);

    for(unsigned long long i = 0; i < possible_strategies; i++)
    {
        csv_file << bitset<8>(i).to_ulong();
        if (i != possible_strategies-1)
            csv_file << ",";
    }
    csv_file << "\n";

    for(unsigned long long i = 0; i < runs; i++)
    {
        for(unsigned long long j = 0; j < possible_strategies; j++)
        {
            csv_file << eta_each_strategy[j][i];
            if (j != possible_strategies-1)
                csv_file << ",";
        }
        csv_file << "\n";
    }
    csv_file.close();
}