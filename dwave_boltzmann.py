#!/home/mircea/.pyenv/versions/3.6.1/bin/python3.6
import numpy as np
import copy
from dimod.reference.samplers import ExactSolver
from neal import SimulatedAnnealingSampler
import random
import math
from collections import namedtuple
from time import sleep

def construct_Q_0():
    Q = dict()

    # Weight construction

    # Fully connection between state and blue nodes
    for i in tuple(range(4)) + tuple(range(12,16)):
        Q[(i,i,)] = 0
        for _ in tuple(range(16)):
            Q[(i,i,)] += ( 2 * random.random() - 1) * random.choice((-1,1,))
        Q[(i,i,)] /= 16

    # Fully connection between action and red nodes
    for i in tuple(range(4,8)) + tuple(range(8,12)):
        Q[(i,i,)] = 0
        for _ in tuple(range(16)):
            Q[(i,i,)] += ( 2 * random.random() - 1) * random.choice((-1,1,))
        Q[(i,i,)] /= 16

    # Weights between red and blue inside cells
    for i,ii in zip(tuple(range(4)),tuple(range(8,12))):
        for j,jj in zip(tuple(range(4,8)),tuple(range(12,16))):
            Q[(i,j)] = 2 * random.random() - 1
            Q[(ii,jj)] = 2 * random.random() - 1

    # Weights between red and blue outside cells
    for i, j in zip(tuple(range(4,8)),tuple(range(12,16))):
        Q[(i,j)] = 2 * random.random() - 1

    return Q

def get_3d_hamiltonian_average_value(samples, Q, replica_count, average_size, big_gamma, beta):
    # samples =\
    #     tuple(
    #         map(
    #             lambda d:\
    #                 tuple(\
    #                     map(\
    #                         lambda ind: d[ind],\
    #                         range(max(d.keys()) + 1),\
    #                     )\
    #                 ),
    #             SimulatedAnnealingSampler().sample_qubo(Q,num_reads=sample_count).samples()
    #         )
    #     )

    i_sample = 0

    h_sum = 0

    w_plus =\
        math.log10(
            math.cosh( big_gamma * beta / replica_count )\
                / math.sinh( big_gamma * beta / replica_count )
        ) / ( 2 * beta )

    for _ in range(average_size):

        new_h_0 = new_h_1 = 0

        j_sample = i_sample

        a = i_sample + replica_count - 1

        while j_sample < a:

            for k_pair, v_weight in Q.items():
                if k_pair[0] == k_pair[1]:
                    new_h_0 = new_h_0 + v_weight * ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )
                else:
                    new_h_0 = new_h_0 + v_weight\
                        * ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )\
                        * ( -1 if samples[j_sample][k_pair[1]] == 0 else 1 )

            for node_index in samples[j_sample].keys():
                new_h_1 = new_h_1\
                    + ( -1 if samples[j_sample][node_index] == 0 else 1 )\
                    * ( -1 if samples[j_sample + 1][node_index] == 0 else 1 )

            j_sample += 1

        for k_pair, v_weight in Q.items():
            if k_pair[0] == k_pair[1]:
                new_h_0 = new_h_0 + v_weight * ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )
            else:
                new_h_0 = new_h_0 + v_weight\
                    * ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )\
                    * ( -1 if samples[j_sample][k_pair[1]] == 0 else 1 )

        for node_index in samples[j_sample].keys():
            new_h_1 = new_h_1\
                + ( -1 if samples[j_sample][node_index] == 0 else 1 )\
                * ( -1 if samples[i_sample][node_index] == 0 else 1 )

        h_sum = h_sum + new_h_0 / replica_count + w_plus * new_h_1

        i_sample += replica_count

    return -1 * h_sum / average_size

def get_free_energy(average_hamiltonina, samples, replica_count, beta):

    key_list = sorted(samples[0].keys())

    prob_dict = dict()

    for i_sample in range(0,len(samples),replica_count):

        c_iterable = list()

        for s in samples[i_sample : i_sample + replica_count]:
            for k in key_list:
                c_iterable.append(s[k])

        c_iterable = tuple(c_iterable)

        if c_iterable in prob_dict:
            prob_dict[c_iterable] += 1
        else:
            prob_dict[c_iterable] = 1

    a_sum = 0

    div_factor = len(samples) // replica_count

    for c in prob_dict.values():
        a_sum = a_sum\
            + c * math.log10( c / div_factor ) / div_factor

    return average_hamiltonina + a_sum / beta

def update_weights(Q_hh, Q_vh, samples, reward, future_F, current_F, visible_iterable,\
    learning_rate, small_gamma):
    '''
    Q_hh contains key pairs (i,j) where i < j

    Q_vh contains key pairs (visible, hidden)
    '''

    prob_dict = dict()

    for s in samples:
        for k_pair in Q_hh.keys():
            if k_pair in prob_dict:
                prob_dict[k_pair] +=\
                    ( -1 if s[k_pair[0]] == 0 else 1 )\
                    * ( -1 if s[k_pair[1]] == 0 else 1 )
            else:
                prob_dict[k_pair] =\
                    ( -1 if s[k_pair[0]] == 0 else 1 )\
                    * ( -1 if s[k_pair[1]] == 0 else 1 )

        for k in s.keys():
            if k in prob_dict:
                prob_dict[k] +=\
                    ( -1 if s[k] == 0 else 1 )
            else:
                prob_dict[k] = ( -1 if s[k] == 0 else 1 )

    for k_pair in Q_hh.keys():
        Q_hh[k_pair] = Q_hh[k_pair] + learning_rate\
            * (reward - small_gamma * future_F + current_F)\
            * prob_dict[k_pair] / len(samples)

    for k_pair in Q_vh.keys():
        Q_vh[k_pair] = Q_vh[k_pair] + learning_rate\
            * (reward - small_gamma * future_F + current_F)\
            * visible_iterable[k_pair[0]]\
            * prob_dict[k_pair[1]] / len(samples)

    return Q_hh, Q_vh

def create_general_Q_from(Q_hh, Q_vh, visible_iterable):

    Q = dict()
    for k_pair, w in Q_hh.items():
        Q[k_pair] = Q[(k_pair[1],k_pair[0],)] = w
    for k_pair, w in Q_vh.items():

        # print(visible_iterable,k_pair[0])

        if (k_pair[1],k_pair[1],) not in Q:
            Q[(k_pair[1],k_pair[1],)] = w * visible_iterable[k_pair[0]]
        else:
            Q[(k_pair[1],k_pair[1],)] += w * visible_iterable[k_pair[0]]
    return Q

def simulate_grid_world_main():
    '''
    link: https://arxiv.org/pdf/1706.00074.pdf

    Action-Set Description
        0 : ^
        1 : >
        2 : v
        3 : <
        4 : hold

    '''
    # Initialize constants

    replica_count = 10
    average_size = 10
    sample_count = replica_count * average_size

    optimal_policy_tuple = (\
        ( (4,) , (3,)   , (3,)    , (3,) , (3,)   ) ,\
        ( (0,) , (3,0,) , tuple() , (0,) , (3,0,) ) ,\
        ( (0,) , (3,0,) , (3,)    , (0,) , (3,0,) ) ,\
    )
    # reward_function_tuple = (\
    #     ( 400 , 100 , 100 , 100 , 100 ) ,\
    #     ( 400 , 100 , 100 , 100 , 100 ) ,\
    #     ( 400 , 100 ,   0 , 100 , 100 ) ,\
    # )
    reward_function_tuple = list()
    for i in range(3):
        reward_function_tuple.append(list())
        for j in range(5):
            reward_function_tuple[-1].append(400 * (3-i+5-j) / 8)

    available_state_dict = dict()
    i = 0
    for q1 in ( ( -1 , ) , ( 1 , ) ,):
        for q2 in ( ( -1 , ) , ( 1 , ) ,):
            for q3 in ( ( -1 , ) , ( 1 , ) ,):
                for q4 in ( ( -1 , ) , ( 1 , ) ,):
                    if i != 7:
                        available_state_dict[ (i//5,i%5,) ] = q1 + q2 + q3 + q4
                    i += 1

    available_actions_list = list()
    i = 0
    for q1 in ( ( -1 , ) , ( 1 , ) ,):
        for q2 in ( ( -1 , ) , ( 1 , ) ,):
            for q3 in ( ( -1 , ) , ( 1 , ) ,):
                if i < 5:
                    available_actions_list.append(q1 + q2 + q3)
                else:
                    break
                i+=1
            if i >= 5:
                break
        if i >= 5:
            break
    available_actions_per_position_tuple = (\
        ( (1,2,4)   , (3,1,2,4) , (3,1,4) , (3,1,2,4) , (3,2,4)   ) ,\
        ( (0,1,2,4) , (3,0,2,4) , tuple() , (0,2,1,4) , (3,0,2,4) ) ,\
        ( (0,1,4)   , (3,0,1,4) , (3,1,4) , (0,3,1,4) , (3,0,4)   ) ,\
    )

    # Initialize weights
    Q_hh = dict()
    for i,ii in zip(tuple(range(4)),tuple(range(8,12))):
        for j,jj in zip(tuple(range(4,8)),tuple(range(12,16))):
            Q_hh[(i,j)] = 2 * random.random() - 1
            Q_hh[(ii,jj)] = 2 * random.random() - 1
    for i, j in zip(tuple(range(4,8)),tuple(range(12,16))):
        Q_hh[(i,j)] = 2 * random.random() - 1

    Q_vh = dict()
        # Fully connection between state and blue nodes
    for j in ( tuple(range(4)) + tuple(range(12,16)) ):
        for i in range(4):
            Q_vh[(i,j,)] = 2 * random.random() - 1
        # Fully connection between action and red nodes
    for j in ( tuple(range(4,8)) + tuple(range(8,12)) ):
        for i in range(4,7):
            Q_vh[(i,j,)] = 2 * random.random() - 1

    def agent_step(current_state):
        max_tuple = None

        for action_index in filter(
                lambda e: e != 4,
                available_actions_per_position_tuple[\
                current_state[0][0]][current_state[0][1]]
            ):

            vis_iterable = current_state[1] + available_actions_list[action_index]

            general_Q = create_general_Q_from(
                Q_hh,
                Q_vh,
                vis_iterable
            )

            samples = SimulatedAnnealingSampler().sample_qubo(
                general_Q,
                num_reads=sample_count
            ).samples()

            current_F = get_free_energy(
                get_3d_hamiltonian_average_value(
                    samples,
                    general_Q,
                    replica_count,
                    average_size,
                    0.5,
                    2
                ),
                samples,
                replica_count,
                2,
            )

            if max_tuple is None or max_tuple[0] < current_F:
                max_tuple = ( current_F , action_index , samples , vis_iterable )

        if max_tuple[1] == 0:
            new_position = (\
                current_state[0][0] - 1,
                current_state[0][1]
            )
        elif max_tuple[1] == 1:
            new_position = (\
                current_state[0][0],
                current_state[0][1] + 1,
            )
        elif max_tuple[1] == 2:
            new_position = (\
                current_state[0][0] + 1,
                current_state[0][1],
            )
        elif max_tuple[1] == 3:
            new_position = (\
                current_state[0][0],
                current_state[0][1] - 1,
            )
        else:
            new_position = current_state[0]

        return\
            (
                (
                    new_position,
                    available_state_dict[new_position],
                ),
                max_tuple[0],
                max_tuple[2],
                max_tuple[3]
            )

    def print_agent(x_y_position):
        for i in range(3):
            line_string = ''
            for j in range(5):
                if x_y_position == (i,j,):
                    line_string += 'X'
                elif (i,j,) == (0,0):
                    line_string += 'G'
                elif (i,j) == (1,2):
                    line_string += 'W'
                else:
                    line_string += 'O'
            print(line_string)
        print()


    for i_round in range(10):

        agent_state_tuple = random.choice(
            tuple(
                filter(
                    lambda e : e[0] != (0,0,),
                    available_state_dict.items()
                )
            )
        )

        step_count = 1
        agent_state_tuple, current_F, current_samples,\
            current_vis_iterable = agent_step(agent_state_tuple)

        if agent_state_tuple[0] != (0,0,):

            while True:

                agent_state_tuple, future_F, future_samples, future_vis_iterable = agent_step(agent_state_tuple)

                if agent_state_tuple[0] == (0,0,):
                    break

                step_count += 1

                Q_hh, Q_vh =\
                    update_weights(
                        Q_hh,
                        Q_vh,
                        current_samples,
                        reward_function_tuple[\
                            agent_state_tuple[0][0]][\
                            agent_state_tuple[0][1]],
                        future_F,
                        current_F,
                        current_vis_iterable,
                        0.5,
                        2
                    )

                current_F, current_samples, current_vis_iterable =\
                    future_F, future_samples, future_vis_iterable

                print(i_round, ':', 'step=', step_count, 'position=', agent_state_tuple[0])
                print_agent(agent_state_tuple[0])

                sleep(1)

        print()



def main_0():
    pass
    # sampleset = get_min_energy_state(
    #         construct_Q_0(),
    #         5
    #     )
    # # print(sampleset, type(sampleset),sampleset.samples())

    # get_3d_hamiltonian_value(
    #     construct_Q_0(),
    #     2,
    #     2,
    #     1,
    #     1
    # )

    simulate_grid_world_main()

    # for s in sampleset.samples():
    #     print(s)


if __name__ == '__main__':
    main_0()