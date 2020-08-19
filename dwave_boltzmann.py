import numpy as np
import copy
from dimod.reference.samplers import ExactSolver
from neal import SimulatedAnnealingSampler
import random
import math
from collections import namedtuple
from time import sleep
import matplotlib.pyplot as plt
import pickle
from agent_utils import *
import multiprocessing
import os
import matplotlib.pyplot as plt

# Paper Link: https://arxiv.org/pdf/1706.00074.pdf

# Bellow different quantities describing the Grid world problem
# are set.
optimal_policy_tuple = (\
    ( (4,) , (3,)   , (3,)    , (3,) , (3,)   ) ,\
    ( (0,) , (3,0,) , tuple() , (0,) , (3,0,) ) ,\
    ( (0,) , (3,0,) , (3,)    , (0,) , (3,0,) ) ,\
)
# reward_function_tuple = (\
#     ( 200 , 100 , 100 , 100 , 100 ) ,\
#     ( 100 , 100 , 100 , 100 , 100 ) ,\
#     ( 100 , 100 ,   0 , 100 , 100 ) ,\
# )
reward_function_tuple = (\
    ( 220 , 200 , 180 , 160 , 140 ) ,\
    ( 200 , 180 , 160 , 120 , 120 ) ,\
    ( 180 , 160 ,   0 , 120 , 100 ) ,\
)
available_state_dict = dict()
i = 0
for q1 in ( ( -1 , ) , ( 1 , ) ,):
    for q2 in ( ( -1 , ) , ( 1 , ) ,):
        for q3 in ( ( -1 , ) , ( 1 , ) ,):
            for q4 in ( ( -1 , ) , ( 1 , ) ,):
                if i != 7:
                    available_state_dict[ (i//5,i%5,) ] = q1 + q2 + q3 + q4
                i += 1
                if i >= 15: break
            if i >= 15: break
        if i >= 15: break
    if i >= 15: break
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
    ( (1,2,4)   , (1,2,3,4) , (1,3,4) , (1,2,3,4) , (2,3,4)   ) ,\
    ( (0,1,2,4) , (0,2,3,4) , tuple() , (0,1,2,4) , (0,2,3,4) ) ,\
    ( (0,1,4)   , (0,1,3,4) , (1,3,4) , (0,1,3,4) , (0,3,4)   ) ,\
)
'''
    Action-Set Description
        0 : ^
        1 : >
        2 : v
        3 : <
        4 : hold
'''


# def agent_step(current_state):
def agent_step(current_state, Q_hh, Q_vh, epsilon_p):
    '''
    Implements a Grid World problem step.
    '''
    max_tuple = None

    if not ( 0 <= current_state[0][0] < 3 and 0 <= current_state[0][1] ):
        print('first debug:',current_state)

    if epsilon_p == 0 or random.random() > epsilon_p:

        # actions_energies_list = list()

        for action_index in filter(
                # lambda e: e != 4,
                lambda e: True,
                available_actions_per_position_tuple[ current_state[0][0] ][ current_state[0][1] ]
            ):

            vis_iterable = current_state[1] + available_actions_list[action_index]

            general_Q = create_general_Q_from(
                Q_hh,
                Q_vh,
                vis_iterable
            )

            samples = list(SimulatedAnnealingSampler().sample_qubo(
                general_Q,
                num_reads=sample_count
            ).samples())

            random.shuffle(samples)

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

            # Uncomment bellow if one wants to log stuff.
            # if action_index == 0:
            #     new_position = (\
            #         current_state[0][0] - 1,
            #         current_state[0][1]
            #     )
            # elif action_index == 1:
            #     new_position = (\
            #         current_state[0][0],
            #         current_state[0][1] + 1,
            #     )
            # elif action_index == 2:
            #     new_position = (\
            #         current_state[0][0] + 1,
            #         current_state[0][1],
            #     )
            # elif action_index == 3:
            #     new_position = (\
            #         current_state[0][0],
            #         current_state[0][1] - 1,
            #     )
            # else:
            #     new_position = current_state[0]

            # actions_energies_list.append((
            #     action_index,
            #     current_F,
            #     reward_function_tuple[new_position[0]][new_position[1]],
            #     vis_iterable
            # ))

            if max_tuple is None or max_tuple[0] < current_F:
                max_tuple = ( current_F , action_index , samples , vis_iterable )

    else:

        # Uncomment bellow if one wants to log stuff.
        # actions_energies_list = list()

        action_index = random.choice(
            tuple(filter(
                # lambda e: e != 4,
                lambda e: True,
                available_actions_per_position_tuple[\
                current_state[0][0]][current_state[0][1]]
            )))

        vis_iterable = current_state[1] + available_actions_list[action_index]

        general_Q = create_general_Q_from(
            Q_hh,
            Q_vh,
            vis_iterable
        )

        samples = list(SimulatedAnnealingSampler().sample_qubo(
            general_Q,
            num_reads=sample_count
        ).samples())

        random.shuffle(samples)

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

        # Uncomment bellow if one wants to log stuff.
        # if action_index == 0:
        #     new_position = (\
        #         current_state[0][0] - 1,
        #         current_state[0][1]
        #     )
        # elif action_index == 1:
        #     new_position = (\
        #         current_state[0][0],
        #         current_state[0][1] + 1,
        #     )
        # elif action_index == 2:
        #     new_position = (\
        #         current_state[0][0] + 1,
        #         current_state[0][1],
        #     )
        # elif action_index == 3:
        #     new_position = (\
        #         current_state[0][0],
        #         current_state[0][1] - 1,
        #     )
        # else:
        #     new_position = current_state[0]

        # actions_energies_list.append((
        #     action_index,
        #     current_F,
        #     reward_function_tuple[new_position[0]][new_position[1]],
        #     vis_iterable
        # ))

        max_tuple = ( current_F , action_index , samples , vis_iterable )

    # Uncomment bellow if one wants to log stuff.
    # print(\
    #     str(current_state[0]) + ' : '\
    #     + str( tuple( map( lambda e: ( e[0] , e[1] ) , actions_energies_list ) ) )
    # )
    # print(\
    #     str(current_state[0]) + ' : '\
    #     + str( tuple( map( lambda e: ( e[0] , e[2] ) , actions_energies_list ) ) )
    # )
    # print(str(current_state[0]) + ' : ')
    # for aaa in range( len( actions_energies_list ) ):
    #     print( '\t' + str( actions_energies_list[aaa][0] ) + ' : ' + str(actions_energies_list[aaa][3]))

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

    # Uncomment bellow if one wants to log stuff.
    # print('choices: ' + str(max_tuple[1]) + ' ' + str(new_position))

    if not ( 0 <= new_position[0] < 3 and 0 <= new_position[1] ):
        print('second debug:',new_position, max_tuple)

    return\
        (
            (
                new_position,
                available_state_dict[new_position],
            ),
            max_tuple[0],     # F value
            max_tuple[2],     # samples
            max_tuple[3],     # used iterable vector
            max_tuple[1],     # action index
            current_state[0], # old_position
        )

def agent_step_1(current_state, Q):
    '''
    Implements a Grid World problem step.
    '''
    sample_dict = list(SimulatedAnnealingSampler().sample_qubo(
            Q,
            num_reads=1
        ).samples())[0]

    a_list = list()
    for k in sorted_qubit_indexes_tuple:
        a_list.append( sample_dict[k] )
    action_index = real_01qbits_to_virtual_qubits_dict[tuple(a_list)]

    if False:
        print( tuple(a_list) )
        print( sorted( real_01qbits_to_virtual_qubits_dict.items() ) )
        exit(0)

    # print(tuple(action_set_to_encoding_tuple_dict.keys()))

    action_index = action_set_to_encoding_tuple_dict[\
            available_actions_per_position_tuple[ current_state[0][0] ][ current_state[0][1] ]\
        ][action_index]


    if action_index == 0:
        new_position = (\
            current_state[0][0] - 1,
            current_state[0][1]
        )
    elif action_index == 1:
        new_position = (\
            current_state[0][0],
            current_state[0][1] + 1,
        )
    elif action_index == 2:
        new_position = (\
            current_state[0][0] + 1,
            current_state[0][1],
        )
    elif action_index == 3:
        new_position = (\
            current_state[0][0],
            current_state[0][1] - 1,
        )
    else:
        new_position = current_state[0]

    return\
        (\
            (\
                new_position,\
                available_state_dict[new_position],\
            ),\
            sample_dict,\
            action_index,\
        )

def print_agent(x_y_position):
    '''
    Prints the agent by postion.

    x_y_position
        Tuple of row and column indexes.
    '''
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

def simulate_grid_world_main():
    '''
    Single core single run of the Grid World problem. Stops
    when the agent reaches the goal position.

    Action-Set Description
        0 : ^
        1 : >
        2 : v
        3 : <
        4 : hold
        5 : ^
        6 : >
        7 : v
    '''
    global Q_hh, Q_vh, sample_count, replica_count, average_size

    # Initialize constants
    replica_count = 10
    average_size = 50
    sample_count = replica_count * average_size
    game_count = 500

    # Modification of reward function.
    # reward_function_tuple = list()
    # for i in range(3):
    #     reward_function_tuple.append(list())
    #     for j in range(5):
    #         reward_function_tuple[-1].append(400 * (2-i+4-j) / 8)
    # for i in range(3):
    #     for j in range(5):
    #         reward_function_tuple[i][j] += 20 * (2-i) / 2


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


    step_count_list = list()

    fidelity_list = list()

    log_f = open('agent_dumps/agent_log.csv','wt',buffering=1)

    log_f.write('starting_x,starting_y,step_count,fidelity\n')

    for i_round in range(game_count):

        agent_state_tuple = random.choice(
            tuple(
                filter(
                    lambda e : e[0] != (0,0,) and e[0] != (1, 2),
                    available_state_dict.items()
                )
            )
        )

        # agent_state_tuple = ( (2,4) , available_state_dict[(2,4)] )

        log_f.write(str(agent_state_tuple[0][0]) + ',' + str(agent_state_tuple[0][1]) + ',')

        print(i_round, ':', 'step =', 0, '; position =', agent_state_tuple[0])
        print_agent(agent_state_tuple[0])

        step_count = 1
        agent_state_tuple, current_F, current_samples,\
            current_vis_iterable, action_index, old_position_tuple = agent_step(agent_state_tuple)
        fidelity_count = 1 if action_index in optimal_policy_tuple[old_position_tuple[0]][old_position_tuple[1]] else 0

        print(i_round, ':', 'step =', step_count, '; position =', agent_state_tuple[0])
        print_agent(agent_state_tuple[0])

        if agent_state_tuple[0] != (0,0,):
            while True:

                agent_state_tuple, future_F, future_samples, future_vis_iterable,\
                    action_index, old_position_tuple = agent_step(agent_state_tuple)

                fidelity_count += 1 if action_index in optimal_policy_tuple[old_position_tuple[0]][old_position_tuple[1]] else 0

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
                        0.0001,
                        0.8
                    )

                if agent_state_tuple[0] == (0,0,):
                    break

                current_F, current_samples, current_vis_iterable =\
                    future_F, future_samples, future_vis_iterable

                print(i_round, ':', 'step =', step_count, '; position =', agent_state_tuple[0])
                print_agent(agent_state_tuple[0])

                # sleep(1)

        fidelity_list.append( fidelity_count/step_count )

        log_f.write(str(step_count) + ',' + str(fidelity_list[-1]) + '\n')

        step_count_list.append(step_count)

        print()

    plt.plot( range(len(step_count_list)) , step_count_list , 'b-' )
    plt.plot( range(len(step_count_list)) , step_count_list , 'ro' )
    # plt.show()
    plt.savefig( 'agent_step_history.png' )

    plt.clf()

    plt.plot( range(len(fidelity_list)) , fidelity_list , 'b-' )
    plt.plot( range(len(fidelity_list)) , fidelity_list , 'ro' )
    # plt.show()
    plt.savefig( 'agent_fidelity_history.png' )

def paper_simulation_main():
    '''
    Single core multiple independent runs if the Grid World problem agent.
    '''

    global Q_hh, Q_vh, sample_count, replica_count, average_size

    runs_count = 1
    actions_count = 2300

    replica_count = 20
    average_size = 50
    sample_count = replica_count * average_size

    fidelity_acc_list = actions_count * [0]

    for run_index in range(runs_count):

        # pickle.dump(
        #     fidelity_acc_list,
        #     open(
        #         './agent_dumps/intermediate_fidelity/fidelity_over_samples_'+str(run_index)+'.p' , 'wb'
        #     )
        # )

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

        agent_state_tuple = random.choice(tuple(available_state_dict.items()))

        agent_state_tuple, current_F, current_samples,\
            current_vis_iterable, action_index, old_position_tuple = agent_step( agent_state_tuple , Q_hh , Q_vh , 0.3 )
        fidelity_acc_list[0] += 1 if action_index in optimal_policy_tuple[old_position_tuple[0]][old_position_tuple[1]] else 0

        for sample_index in range(1,actions_count):

            agent_state_tuple, future_F, future_samples, future_vis_iterable,\
                action_index, old_position_tuple = agent_step(\
                    agent_state_tuple,\
                    Q_hh,\
                    Q_vh,\
                    0.9 if sample_index < 900 else ( 0.9 * ( actions_count - sample_index ) / ( actions_count - 900 ) ),\
                )

            fidelity_acc_list[sample_index] += 1 if action_index in optimal_policy_tuple[old_position_tuple[0]][old_position_tuple[1]] else 0

            print_agent( old_position_tuple )
            # sleep(1)

            Q_hh, Q_vh =\
            update_weights(
                Q_hh,
                Q_vh,
                current_samples,
                reward_function_tuple[\
                    agent_state_tuple[0][0]][\
                    agent_state_tuple[0][1]],
                0  ,
                current_F,
                current_vis_iterable,
                0.001 if sample_index < 150 else\
                    ( 0.001 - 0.0009 * ( sample_index - 150 ) / ( actions_count - 150 ) ),
                0.8,
            )

            current_F, current_samples, current_vis_iterable =\
                future_F, future_samples, future_vis_iterable

    pickle.dump(
        fidelity_acc_list,
        open( './agent_dumps/fidelity_over_samples.p' , 'wb' )
    )

def simulate_independent_run(index):
    '''
    Implents indepent run to be processed on an independet core.
    '''
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

    fidelity_f = open('./agent_fidelity_dump/' + str(index) + '.txt','wt')

    # print( 'Q_hh' , len(tuple(Q_hh.keys())) ) Q_hh 36
    # print( 'Q_vh' , len(tuple(Q_vh.keys())) ) Q_vh 56

    agent_state_tuple = random.choice(tuple(available_state_dict.items()))

    agent_state_tuple, current_F, current_samples,\
        current_vis_iterable, action_index, old_position_tuple = agent_step(agent_state_tuple, Q_hh, Q_vh,0.9)

    if action_index in optimal_policy_tuple[old_position_tuple[0]][old_position_tuple[1]]:

        shared_fidelity_array_list[ index ][ 0 ] = 1

        fidelity_f.write( '1,' )

    else:

        fidelity_f.write( '0,' )

    fidelity_f.flush()

    a_semaphore.release()

    for sample_index in range(1,actions_count):

        agent_state_tuple, future_F, future_samples, future_vis_iterable,\
            action_index, old_position_tuple = agent_step(\
                agent_state_tuple,\
                Q_hh,\
                Q_vh,\
                0.9 if sample_index < 900 else ( 0.9 * ( actions_count - sample_index ) / ( actions_count - 900 ) ),\
        )

        if action_index in optimal_policy_tuple[old_position_tuple[0]][old_position_tuple[1]]:

            shared_fidelity_array_list[ index ][ sample_index ] = 1

            fidelity_f.write( '1,' )

        else:

            fidelity_f.write( '0,' )

        fidelity_f.flush()

        # print( index , ' : ' , action_index ,\
        #     optimal_policy_tuple[old_position_tuple[0]][old_position_tuple[1]] ,\
        #     sum( shared_fidelity_array_list[index] ) / sample_index
        # )

        a_semaphore.release()

        # print( index , 1 if action_index in optimal_policy_tuple[old_position_tuple[0]][old_position_tuple[1]] else 0 )

        Q_hh,Q_vh=update_weights(
            Q_hh,
            Q_vh,
            current_samples,
            reward_function_tuple[\
                agent_state_tuple[0][0]][\
                agent_state_tuple[0][1]],
            0,
            current_F,
            current_vis_iterable,
            0.001 if sample_index < 150 else\
                ( 0.001 - 0.0009 * ( sample_index - 150 ) / ( actions_count - 150 ) ),
            0.8,
        )

        # if sample_index % 10 == 0:
        #     print( index , ':' , sample_index , sum( shared_fidelity_array_list[index] ) / sample_index )

        current_F, current_samples, current_vis_iterable =\
            future_F, future_samples, future_vis_iterable

def partial_results_dump():
    '''
    Runs in a separate process and dumps to disk the fidelity
    at each time step from a shared array with the independent
    run processes.
    '''
    with open( 'grid_world_fidelity.txt' , 'wt' ) as fidelity_file:

        for t in range( actions_count ):

            for _ in range(independent_run_count): a_semaphore.acquire()

            fidelity_file.write(
                str(
                    sum( map( lambda arr: arr[ t ] , shared_fidelity_array_list ) ) / independent_run_count
                ) + ','
            )

            fidelity_file.flush()

def paper_simulation_main_1():
    '''
    Parallel simulation of multiple independt runs.
    '''
    global shared_fidelity_array_list, a_semaphore
    global sample_count, replica_count, average_size, actions_count, independent_run_count

    independent_run_count = 38

    actions_count = 10000

    replica_count = 5
    average_size = 25
    sample_count = replica_count * average_size

    shared_fidelity_array_list = list()

    a_semaphore = multiprocessing.Semaphore(independent_run_count)

    for _ in range( independent_run_count ):

        shared_fidelity_array_list.append(
            multiprocessing.RawArray(
                'i',
                actions_count * [0]
            )
        )

        a_semaphore.acquire()

    dump_proc = multiprocessing.Process(\
        target=partial_results_dump\
    )

    dump_proc.start()

    multiprocessing.Pool(independent_run_count).map(\
        simulate_independent_run,
        range( independent_run_count )
    )

    dump_proc.join()

    a = actions_count * [0]
    for arr in shared_fidelity_array_list:
        for i in range( actions_count ):
            a[i] += arr[i]

    pickle.dump(
        tuple(map(lambda e: e / independent_run_count , a )),
        open( 'pipe_fidelity.p' , 'wb' )
    )

def policy_train_per_proc(index):
    Q_hh = dict()
    for i in (0,1,):
        for j in (4,5,):
            Q_hh[(i,j)] = 2 * random.random() - 1
    Q_vh = dict()
    for i in range(4):
        for j in (0,1,4,5,):
            Q_vh[(i,j)] = 2 * random.random() - 1

    agent_state_tuple = random.choice(tuple(available_state_dict.items()))

    old_agent_state_tuple, old_action_real_qbits_dict, old_action_index = agent_step_1(
        agent_state_tuple,
        create_general_Q_from(Q_hh, Q_vh, agent_state_tuple[1])
    )

    if old_action_index in optimal_policy_tuple[old_agent_state_tuple[0][0]][old_agent_state_tuple[0][1]]:
        lock_list[0].acquire()
        fidelity_array[0] += 1
        lock_list[0].release()

    for t in range(1,steps_count):
        new_agent_state_tuple, new_action_real_qbits_dict, new_action_index = agent_step_1(
            old_agent_state_tuple,
            create_general_Q_from(Q_hh, Q_vh, old_agent_state_tuple[1])
        )

        Q_hh, Q_vh = update_weights_1(\
            old_agent_state_tuple[1],\
            Q_hh,\
            Q_vh,\
            reward_function_tuple[agent_state_tuple[0][0]][agent_state_tuple[0][1]],\
            old_action_real_qbits_dict,\
            0.6,\
            0.00001,\
        )

        old_agent_state_tuple, old_action_real_qbits_dict, old_action_index =\
            new_agent_state_tuple, new_action_real_qbits_dict, new_action_index

        if old_action_index in optimal_policy_tuple[old_agent_state_tuple[0][0]][old_agent_state_tuple[0][1]]:
            lock_list[t].acquire()
            fidelity_array[t] += 1
            lock_list[t].release()

    print('Finished run',index,'!')

def policy_gradient_train():
    '''
    Policy agent attempt. Needs more looking into.
    '''
    global action_set_to_encoding_tuple_dict,\
        real_01qbits_to_virtual_qubits_dict, fidelity_array,\
        lock_list, steps_count, sorted_qubit_indexes_tuple

    sorted_qubit_indexes_tuple = ( 0 , 1 , 4 , 5 , )

    all_possible_action_encodings_tuple = get_all_possible_configurations(
        tuple() , 0 , 3 , (-1,1)
    )

    real_01qbits_to_virtual_qubits_dict = dict()
    for q1,q2,q3,q4 in get_all_possible_configurations(tuple(), 0, 4, (0,1)):
        real_01qbits_to_virtual_qubits_dict[(q1,q2,q3,q4,)] = (\
            -1 if q1 == 0 else 1,\
            -1 if (q2 + q4) % 2 == 0 else 1,\
            -1 if q3 == 0 else 1,\
        )

    action_set_to_encoding_tuple_dict = dict()
    for actions_line in available_actions_per_position_tuple:
        for actions_tuple in actions_line:
            if len(actions_tuple) > 0 and actions_tuple not in action_set_to_encoding_tuple_dict:
                action_set_to_encoding_tuple_dict[actions_tuple] = dict()
                i = 0
                for q_bit_enc_tuple in all_possible_action_encodings_tuple:
                    action_set_to_encoding_tuple_dict[actions_tuple][q_bit_enc_tuple] = actions_tuple[i]
                    i = ( i + 1 ) % len(actions_tuple)

    runs_count = 100
    steps_count = 500

    fidelity_array = multiprocessing.RawArray('i',steps_count*[0])
    lock_list = list()
    for _ in range( steps_count ):
        lock_list.append( multiprocessing.Lock() )

    multiprocessing.Pool( 7 ).map(policy_train_per_proc , range(runs_count))
    # policy_train_per_proc(0)

    pickle.dump(
        tuple( map( lambda e: e / runs_count, fidelity_array ) ),
        open( './agent_dumps/reinforce_agent_fidelity.p','wb' )
    )

def plot_agent_fidelity_from_upb():
    '''
    Was used to get fidelity from a remote computer.
    '''
    print('Input a valid user and remote host and comment this !')
    exit(0)

    os.system(
        'scp remote_user@remote_host:./grid_world_fidelity.txt .'
    )

    with open( 'grid_world_fidelity.txt' ,'rt' ) as f:
        a =\
            tuple(\
                map(\
                    lambda e: float(e),\
                    f.read().split(',')[:-1]\
                )
            )
        plt.plot(range(len(a)),a,label='Fidelity')
        plt.legend()
        plt.show()

def test_main_0():
    '''
    Tests the updating mechanism. The free energies should go
    towards the values in "reward_tuple".
    '''
    global replica_count,\
        average_size,\
        sample_count

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

    replica_count = 3
    average_size = 5
    sample_count = replica_count * average_size

    reward_tuple =\
        (\
            500,\
            400,\
            300,\
            200,\
            100,\
        )

    tries_count = 3000

    agent_state_tuple = random.choice(tuple(available_state_dict.items()))

    f_value_list = list()

    for _ in range( tries_count ):

        f_value_list.append( list() )

        for action_index in range(5):

            vis_iterable = agent_state_tuple[1] + available_actions_list[action_index]

            general_Q = create_general_Q_from(
                Q_hh,
                Q_vh,
                vis_iterable
            )

            samples = list(SimulatedAnnealingSampler().sample_qubo(
                general_Q,
                num_reads=sample_count
            ).samples())

            random.shuffle(samples)

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

            Q_hh, Q_vh =\
                update_weights(
                    Q_hh,
                    Q_vh,
                    samples,
                    reward_tuple[action_index],
                    0,
                    current_F,
                    vis_iterable,
                    0.0001,
                    0.8
                )

            f_value_list[-1].append( current_F )

        # print( f_value_list[-1] )

    plt.plot(
        range(len(f_value_list)),
        tuple(map(lambda e: e[0],f_value_list)),
        label='First'
    )
    plt.plot(
        range(len(f_value_list)),
        tuple(map(lambda e: e[1],f_value_list)),
        label='Second'
    )
    plt.plot(
        range(len(f_value_list)),
        tuple(map(lambda e: e[2],f_value_list)),
        label='Third'
    )
    plt.plot(
        range(len(f_value_list)),
        tuple(map(lambda e: e[3],f_value_list)),
        label='Fourth'
    )
    plt.plot(
        range(len(f_value_list)),
        tuple(map(lambda e: e[4],f_value_list)),
        label='Fifth'
    )
    plt.legend()
    plt.show()

if __name__ == '__main__':

    # main_0()
    # paper_simulation_main_1()
    # policy_gradient_train()

    # paper_simulation_main()

    test_main_0()

    # import sys
    # if sys.argv[1] == '0':
    #     paper_simulation_main_1()
    # elif sys.argv[1] == '1':
    #     plot_agent_fidelity_from_upb()
    # else:
    #     a = pickle.load(open('./pipe_fidelity.p','rb'))
    #     plt.plot(range(len(a)),a)
    #     plt.show()