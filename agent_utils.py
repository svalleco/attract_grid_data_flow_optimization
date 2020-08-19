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

def get_3d_hamiltonian_average_value(samples, Q, replica_count, average_size, big_gamma, beta):
	'''
	It produces the average Hamiltonian of one dimension higher.

	samples
		It is a list containg the samples from the DWAVE API.

	Q
		It is a dict containg the weights of the Chimera graph.

	replica_count
		It contains the number of replicas in the Hamiltonian of one dimension higher.

	average_size
		It contains the number of configurations of the Hamiltonian of one dimension higher
		used for extracting the value.

	big_gamma, beta
		The parameters with the signification given in the paper.
	'''
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

			added_set = set()

			for k_pair, v_weight in Q.items():

				if k_pair[0] == k_pair[1]:

					new_h_0 = new_h_0 + v_weight * ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )

				else:

					if k_pair not in added_set and ( k_pair[1] , k_pair[0] , ) not in added_set:
					# if True:

						new_h_0 = new_h_0 + v_weight\
							* ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )\
							* ( -1 if samples[j_sample][k_pair[1]] == 0 else 1 )

						added_set.add( k_pair )

			for node_index in samples[j_sample].keys():

				new_h_1 = new_h_1\
					+ ( -1 if samples[j_sample][node_index] == 0 else 1 )\
					* ( -1 if samples[j_sample + 1][node_index] == 0 else 1 )

			j_sample += 1

		added_set = set()

		for k_pair, v_weight in Q.items():

			if k_pair[0] == k_pair[1]:

				new_h_0 = new_h_0 + v_weight * ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )

			else:

				if k_pair not in added_set and ( k_pair[1] , k_pair[0] , ) not in added_set:
				# if True:

					new_h_0 = new_h_0 + v_weight\
						* ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )\
						* ( -1 if samples[j_sample][k_pair[1]] == 0 else 1 )

					added_set.add( k_pair )


		for node_index in samples[j_sample].keys():

			new_h_1 = new_h_1\
				+ ( -1 if samples[j_sample][node_index] == 0 else 1 )\
				* ( -1 if samples[i_sample][node_index] == 0 else 1 )

		h_sum = h_sum + new_h_0 / replica_count + w_plus * new_h_1

		i_sample += replica_count

	return -1 * h_sum / average_size

def get_free_energy(average_hamiltonina, samples, replica_count, beta):
	'''
	It calculates the free energy after the formula in the paper.

	average_hamiltonina
		It is the value of the average of the Hamiltonians of one dimension higher. It is
		created by calling "get_3d_hamiltonian_average_value".

	samples
		It is created by calling the DWAVE API.

	replica_count
		It is the number of replicas in the Hamiltonian of one dimension higher.

	beta
		Parameter presented in the paper.
	'''

	key_list = sorted(samples[0].keys())

	prob_dict = dict()

	for i_sample in range(0,len(samples),replica_count):

		c_iterable = list()

		for s in samples[i_sample : i_sample + replica_count]:
			for k in key_list:
				c_iterable.append( s[k] )

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
	Q_hh
		Contains key pairs (i,j) where i < j

	Q_vh
		Contains key pairs (visible, hidden)

	samples
		It is created by calling the DWAVE API.

	reward
		It is the reward that from either the Environment Network or
		directly from MonALISA.

	future_F
		It is the reward the agent gets at moment t + 1.

	current_F
		It is the reward the agent gets at moment t.

	visible_iterable
		It is the visible units -1/1 iterable the agent uses at moment t.

	learning_rate
		It is the learning rate used in the TD(0) algorithm.

	small_gamma
		It is the discount factor used in the TD(0) algorithm.
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
		Q_hh[k_pair] = Q_hh[k_pair] - learning_rate\
			* ( reward + small_gamma * future_F - current_F )\
			* prob_dict[k_pair] / len(samples)

	for k_pair in Q_vh.keys():
		Q_vh[k_pair] = Q_vh[k_pair] - learning_rate\
			* ( reward + small_gamma * future_F - current_F )\
			* visible_iterable[k_pair[0]]\
			* prob_dict[k_pair[1]] / len(samples)

	return Q_hh, Q_vh

def create_general_Q_from(Q_hh, Q_vh, visible_iterable):
	'''
	Creates a weight dict that can be used with the DWAVE API. As the
	visible units are clamped, they are incorporated into biases.

	Q_hh
		Contains key pairs (i,j) where i < j for hidden-hidden weights.

	Q_vh
		Contains key pairs (visible, hidden) for visible-hidden weights.

	visible_iterable
		Contains -1/1 values.
	'''
	Q = dict()

	for k_pair, w in Q_hh.items():
		Q[k_pair] = Q[(k_pair[1],k_pair[0],)] = w

	for k_pair, w in Q_vh.items():

		if (k_pair[1],k_pair[1],) not in Q:
			Q[(k_pair[1],k_pair[1],)] = w * visible_iterable[k_pair[0]]
		else:
			Q[(k_pair[1],k_pair[1],)] += w * visible_iterable[k_pair[0]]

	return Q

def env_agent_step(\
		current_state,\
		available_actions_per_position_tuple,\
		available_actions_list,\
		Q_hh,\
		Q_vh,\
		replica_count,\
		average_size,\
		apply_action_function\
	):
	'''
	It implements an agent step in the context of the Grid World agent.

	current_state
		It is a tuple containing at index 0 the row and column index of the agent. At
		index 1, it contains the qubit representation of that position.

	available_actions_per_position_tuple
		It contains the action index for the Grid World agent.
        0 : ^
        1 : >
        2 : v
        3 : <
        4 : hold

	available_actions_list
		It contains the actions qubit encoding.

	Q_hh
		It is a dictionary where key pairs (i,j) where i < j are assigned hidden-hidden weights.

	Q_vh
		It is a dictionary where key pairs are assigned visible-hidden weights.

	replica_count
		It contains the number of replicas in the Hamiltonian of one dimension higher.

	average_size
		It contains the number of configurations of the Hamiltonian of one dimension higher
		used for extracting the value.


	'''
	max_tuple = None

	if not ( 0 <= current_state[0][0] < 3 and 0 <= current_state[0][1] ):
		print('first debug:',current_state)

	for action_index in filter(
			# lambda e: e != 4,
			lambda e: True,
			available_actions_per_position_tuple[\
			current_state[0][0]][current_state[0][1]]
		):

		vis_iterable = current_state[1] + available_actions_list[action_index]

		general_Q = create_general_Q_from(
			Q_hh,
			Q_vh,
			vis_iterable
		)

		samples = list(SimulatedAnnealingSampler().sample_qubo(
			general_Q,
			num_reads=replica_count * average_size
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

		if max_tuple is None or max_tuple[0] < current_F:
			max_tuple = ( current_F , action_index , samples , vis_iterable )

	return\
		(
			apply_action_function(), # new state
			max_tuple[0],     		 # F value
			max_tuple[2],            # samples
			max_tuple[3],            # used iterable vector
			max_tuple[1],            # action index
			current_state[0],        # old_position
		)

def get_levels_encodings(qbits_number):
	'''
	It splits the interval [-1 ; +1] into equally sized bins. It returns a list of
	tuples. Each tuple contains:
		inferior limit of the bin
		superior limit of the bin
		encoding in a -1/1 iterable
		middle float value of the bin

	qbits_number
		Number of available qubits (e.g. 5 qubits --> 32 bins)
	'''
	diff = 2 / (2**qbits_number)

	def get_qbits_encodings(previous_tuple, remaining_qb_n):
		if remaining_qb_n == 1:
			return ( previous_tuple + (-1,) , previous_tuple + (1,) , )
		return\
			get_qbits_encodings( previous_tuple + (-1,) , remaining_qb_n - 1 )\
			+ get_qbits_encodings( previous_tuple + (1,) , remaining_qb_n - 1 )

	a_list = list()

	i = -1

	for e in get_qbits_encodings(tuple(), qbits_number):

		a_list.append( ( i , i + diff , e , ( i + i + diff ) / 2 , ) )

		i+=diff

	a_list[-1] = ( a_list[-1][0], 1.1 , a_list[-1][2] , a_list[-1][-1] )

	return a_list

def create_ecoding(float_number, interval_to_qbit_tuple):
	'''
	Does binary search to assign a bin to the input "float_number". Basically
	translates from a float number to a qubit encoding.

	Returns:
		(
			0 -> inferior limit,
			1 -> superior limit,
			2 -> encoding iterable with -1 and 1,
			3 -> value between (-1 and 1),
			4 -> level,
		)

	float_number
		It is the float number to be encoded.

	interval_to_qbit_tuple
		It is a list containing bin information resulted from the
		"get_levels_encodings" function.
	'''

	if interval_to_qbit_tuple[0][0] <= float_number < interval_to_qbit_tuple[0][1]:
		return interval_to_qbit_tuple[0] + ( 0 , )
	if interval_to_qbit_tuple[-1][0] <= float_number < interval_to_qbit_tuple[-1][1]:
		return interval_to_qbit_tuple[-1] + ( len(interval_to_qbit_tuple) - 1 , )

	a = 0

	b = len(interval_to_qbit_tuple) - 1

	while True:
		if b == a + 2:
			if interval_to_qbit_tuple[a][0] <= float_number < interval_to_qbit_tuple[a][1]:
				return interval_to_qbit_tuple[a] + ( a , )
			if interval_to_qbit_tuple[a+1][0] <= float_number < interval_to_qbit_tuple[a+1][1]:
				return interval_to_qbit_tuple[a+1] + ( a + 1 , )
			return interval_to_qbit_tuple[b] + ( b , )
		m = (b + a) // 2
		if interval_to_qbit_tuple[m][0] <= float_number < interval_to_qbit_tuple[m][1]:
			return interval_to_qbit_tuple[m] + ( m , )
		if float_number < interval_to_qbit_tuple[m][0]:
			b = m
		else:
			a = m

def from_qubits_to_number(qubit_iterable):
	'''
	Translates from a qubit encoding an integer one.
	'''
	return\
		sum(
			map(
				lambda a: 0 if a[1] == -1 else 2**a[0],
				map(
					lambda b: (len(qubit_iterable) - 1 - b[0], b[1],),
					enumerate( qubit_iterable )
				)
			)
		)

def from_number_to_qubits(decimal_n, representation_length):
	'''
	Encodes an integer to a qubit interable.

	decimal_n
		The decimal number to be translated.

	representation_length
		How many qubits to use
	'''
	a_list = list()
	for i in range(representation_length - 1, -1, -1):
		if 2**i > decimal_n:
			a_list.append( -1 )
		else:
			a_list.append( 1 )
			decimal_n -= 2**i
	return tuple(a_list)

def get_possible_actions_on_state_func_0(\
	current_state,\
	partition_possible_offsets_iterable,\
	small_partition_length,\
	big_partition_start,\
	small_partitions_count,\
	max_limit_per_small_partition,\
	max_limit_per_big_partition,\
	action_enc_dict):
	'''
	It was once used in a "partition approach" such that the agent
	could modify all the components. Normally, if there are 3
	possible actions per component (up/hold/down one level) and
	there are 11 components, then there could be up to 3^11 possible
	actions in a state ! One possible solutions was to try to
	modify the components in bulk, to extend the "levels" concept
	to more than one component. For example, if there were
	55 qubits in total for all the 11 components and 3 partitions,
	then the partitions would look like"
		first partition
			small partition - 18 qubits
			qubits in the index interval [ 0 , 18 )
		second partiton
			small partition - 18 qubits
			qubits in the index interval [18 , 36 )
		third partiton
			big partition - 19 qubits
			qubits in the index interval [36 , 55 )
	In the aforementioned exameple the maximum action space
	would be down to 3*3*3.
	This approach was eventually dropped, because it would
	vary the component values too much.

	current_state
		Iterable containing the -1/1 encoding of the principal components.

	partition_possible_offsets_iterable
		Iterable containig the possible offsets (usually it is (-1,0,+1)).

	small_partition_length
		Number of qubits that make up a small partition

	big_partition_start
		The index at which the big partition starts

	small_partitions_count
		The count of small partitions

	max_limit_per_small_partition
		The superior integer limit of a small partition

	max_limit_per_big_partition
		The superior integer limit of the big partition

	action_enc_dict
		Dictionary that maps a actions offsets encoding to a -1/1 ecoding
	'''
	available_actions_per_partition_list = list()

	i = 0
	for _ in range(small_partitions_count):
		available_actions_per_partition_list.append(
			tuple(
				filter(
					lambda offset: 0 <= from_qubits_to_number(\
						current_state[i:i+small_partition_length])\
						+ offset < max_limit_per_small_partition,
					partition_possible_offsets_iterable
				)
			)
		)

		i+=small_partition_length

	available_actions_per_partition_list.append(
		tuple(
			filter(
				lambda offset: 0 <= from_qubits_to_number(\
					current_state[big_partition_start:]) + offset < max_limit_per_big_partition,
				partition_possible_offsets_iterable
			)
		)
	)

	def get_tuple_of_actions_for_encoding(acc, i):
		if i == small_partitions_count:
			a_list = list()
			for av_act in available_actions_per_partition_list[-1]:
				a_list.append( acc + (av_act,) )
			return a_list
		a_list = list()
		for av_act in available_actions_per_partition_list[i]:
			a_list += get_tuple_of_actions_for_encoding(\
				acc + (av_act,),\
				i+1,\
			)
		return a_list

	return\
		tuple(
			map(
				lambda e: action_enc_dict[e],
				get_tuple_of_actions_for_encoding(tuple(),0,)
			)
		)

def get_possible_actions_on_state_func_1(\
	current_state,\
	partition_possible_offsets_iterable,\
	q_bits_num,\
	comp_count,\
	max_v,\
	action_enc_dict):
	'''
	It was once used in a "partition approach" such that the agent
	could modify all the components. Normally, if there are 3
	possible actions per component (up/hold/down one level) and
	there are 11 components, then there could be up to 3^11 possible
	actions in a state ! One possible solutions was to try to
	modify the components in bulk, to extend the "levels" concept
	to more than one component. For example, if there were
	55 qubits in total for all the 11 components and 3 partitions,
	then the partitions would look like"
		first partition
			small partition - 18 qubits
			qubits in the index interval [ 0 , 18 )
		second partiton
			small partition - 18 qubits
			qubits in the index interval [18 , 36 )
		third partiton
			big partition - 19 qubits
			qubits in the index interval [36 , 55 )
	In the aforementioned exameple the maximum action space
	would be down to 3*3*3.
	This approach was eventually dropped, because it would
	vary the component values too much.
	'''

	available_actions_per_partition_list = list()

	for i in range( q_bits_num , q_bits_num * ( comp_count + 1 ), q_bits_num ):
		available_actions_per_partition_list.append(
			tuple(
				filter(
					lambda offset: 0 <= from_qubits_to_number(\
						current_state[i:i+q_bits_num]) + offset < max_v,
					partition_possible_offsets_iterable #( -1 , 0 , 1 )
				)
			)
		)

	def get_tuple_of_actions_for_encoding(acc, i):
		if i == comp_count:
			a_list = list()
			for av_act in available_actions_per_partition_list[-1]:
				a_list.append( acc + (av_act,) )
			return a_list
		a_list = list()
		for av_act in available_actions_per_partition_list[i]:
			a_list += get_tuple_of_actions_for_encoding(\
				acc + (av_act,),\
				i+1,\
			)
		return a_list

	return\
		tuple(
			map(
				lambda e: action_enc_dict[e],
				get_tuple_of_actions_for_encoding(tuple(),0,)
			)
		)

def get_possible_actions_on_state_func_2(\
		current_state,\
		partition_possible_offsets_iterable,\
		comp_count,\
		max_v,\
		action_enc_dict):
	'''
	It was modified from "get_possible_actions_on_state_func_1" to work
	on the first 3 components instead of partitions.

	current_state
		A "State" object as defined in "agent_main_2.py"

	partition_possible_offsets_iterable
		Iterable containig the possible offsets (usually it is (-1,0,+1)).

	comp_count
		Count of modifiable components (e.g. 3)

	max_v
		Maximum value of a component

	action_enc_dict
		Dictionary that maps a tuple of integer actions to a qubit
		encoding.
	'''
	available_actions_per_partition_list =\
	tuple(\
		map(
			lambda comp_level:\
				tuple(\
					filter(\
						lambda act_offset: 0 <= comp_level + act_offset < max_v,\
						partition_possible_offsets_iterable\
					)\
				),\
			current_state.modifiable_pc_levels\
		)\
	)

	a = comp_count - 1

	def get_tuple_of_actions_for_encoding(acc, i):
		if i == a:
			a_list = list()
			for av_act in available_actions_per_partition_list[-1]:
				a_list.append( acc + (av_act,) )
			return a_list
		a_list = list()
		for av_act in available_actions_per_partition_list[i]:
			a_list += get_tuple_of_actions_for_encoding(\
				acc + (av_act,),\
				i+1,\
			)
		return a_list

	return\
		tuple(
			map(
				lambda e: (e,action_enc_dict[e],),
				get_tuple_of_actions_for_encoding(tuple(),0,)
			)
		)

def apply_action_func(\
	action_enc,\
	current_state,\
	small_partitions_length,\
	small_partitions_count,\
	big_partition_start,\
	enc_to_tuple_dict,\
	):
	'''
	Used in the partition approach in order to apply a chosen action
	to the current state qubit encoding.
	'''
	action_tuple = enc_to_tuple_dict[action_enc]

	new_state = list()

	i = 0
	for i_partition in range(small_partitions_count):

		new_state += from_number_to_qubits(\
			from_qubits_to_number( current_state[i:i+small_partitions_length] ) + action_tuple[i_partition],
			small_partitions_length,
		)

		i += small_partitions_length

	new_state += from_number_to_qubits(\
		from_qubits_to_number( current_state[big_partition_start:] ) + action_tuple[-1],
		len(current_state[big_partition_start:]),
	)

	return new_state

def apply_action_func_1(\
	action_enc,\
	current_state,\
	q_bits_num,\
	comp_count,\
	enc_to_tuple_dict,\
	new_remainig_components
	):
	'''
	Used in the partition approach in order to apply a chosen action
	to the current state qubit encoding.
	'''
	action_tuple = enc_to_tuple_dict[action_enc]

	new_state = tuple()

	for j,i in enumerate(range( q_bits_num , q_bits_num * ( comp_count + 1 ), q_bits_num )):

		new_state += from_number_to_qubits(\
			from_qubits_to_number( current_state[i:i+q_bits_num] ) + action_tuple[j],
			q_bits_num,
		)

	return new_state + new_remainig_components

def get_reward_through_model(pc_encs_list, ars, keras_model, qb_per_pc_count, interval_to_qbit_tuple,\
	tr_l):
	'''
	It returns the reward by querying a keras model. It gets a lock, because if multiple runs
	work in parallel, tensorflow sessions might block. So the lock manages the access to inference.

	pc_encs_list
		Encodings of the components

	ars
		Encoding of the average read size

	keras_model
		The keras model

	qb_per_pc_count
		The number of qubits used per component

	interval_to_qbit_tuple
		Tuple containig tuples of:
			inferior float limit
			superior float limit
			-1/+1 encoding
	'''
	a_list = list()

	for t in range(40):
		a_list.append(list())

		for a,b,c in interval_to_qbit_tuple:
			if c == ars:
				a_list[-1].append( ( a + b ) / 2 )
				break

		for i in range(0, len(pc_encs_list), qb_per_pc_count):
			for a,b,c in interval_to_qbit_tuple:
				if c == tuple( pc_encs_list[i:i+qb_per_pc_count] ):
					a_list[-1].append( ( a + b ) / 2 )
					break

	tr_l.acquire()

	a = keras_model.predict(np.array([a_list,]))[0]

	tr_l.release()

	return a

def get_reward_through_model_2(pc_encs_list, ars, qb_per_pc_count, interval_to_qbit_tuple,\
	run_index, input_raw_array, output_raw_array, signal_q, result_ready_lock, work_lock):
	'''
	It returns the reward by querying a keras model. It implements the same functionality
	as in "get_reward_through_model". It comunicates with an inference process.
	'''

	for t in range( 0 , 480 , 12 ):

		for a,b,c in interval_to_qbit_tuple:
			if c == ars:
				input_raw_array[t] = ( a + b ) / 2
				break

		for i,j in zip(range(0, len(pc_encs_list), qb_per_pc_count),range(t+1,t+12)):
			for a,b,c in interval_to_qbit_tuple:
				if c == tuple( pc_encs_list[i:i+qb_per_pc_count] ):
					input_raw_array[j] = ( a + b ) / 2
					break

	work_lock.acquire()

	signal_q.put( run_index )

	result_ready_lock.acquire()

	return output_raw_array[run_index]

def get_reward_through_model_1(\
	pc_encs_list,\
	ars,\
	keras_model_path,\
	qb_per_pc_count,
	interval_to_qbit_tuple):
	'''
	DEPRECATED
	'''
	from os import system

	a_list = list()

	for t in range(40):
		a_list.append(list())

		for a,b,c in interval_to_qbit_tuple:
			if c == ars:
				a_list[-1].append( ( a + b ) / 2 )
				break

		for i in range(0, len(pc_encs_list), qb_per_pc_count):
			for a,b,c in interval_to_qbit_tuple:
				if c == tuple( pc_encs_list[i:i+qb_per_pc_count] ):
					a_list[-1].append( ( a + b ) / 2 )
					break

	pickle.dump(
		[ a_list , ],
		open('pipe.p','wb')
	)

	system(
		'python3.7 get_reward.py'
	)

	return pickle.load(open('pipe.p','rb'))

def get_all_possible_configurations(acc, i, q_count, qubits_domain):
	'''
	Returns all possible configurations as a tuple of tuples of length
	q_count.

	acc
		Accumulator

	i
		Current qubit index. Starts at 0.

	q_count
		Number of qubits

	qubits_domain
		An iterable of length 2 (e.g. (0,1,) or (-1,+1,))
	'''
	if i == q_count - 1:
		return\
			(\
				acc + (qubits_domain[0],),\
				acc + (qubits_domain[1],),\
			)
	return\
		get_all_possible_configurations(\
			acc + ( qubits_domain[0] , ) , i + 1 , q_count , qubits_domain )\
		+ get_all_possible_configurations(\
			acc + ( qubits_domain[1] , ) , i + 1 , q_count , qubits_domain )

def update_weights_1(old_state_encoding_tuple, Q_hh, Q_vh, reward, real_qubits_dict, gamma, lr):
	'''
	Policy based update of weights. It has not been extensively tested.
	'''
	minus_pi = 0
	for pair, w_hh in Q_hh.items():
		minus_pi += w_hh * (2 * real_qubits_dict[pair[0]] - 1)\
			* (2*real_qubits_dict[pair[1]] - 1)
	for pair, w_vh in Q_vh.items():
		minus_pi += w_vh * old_state_encoding_tuple[pair[0]]\
			* (2*real_qubits_dict[pair[1]] - 1)
	minus_pi = - minus_pi

	for pair in Q_hh.keys():
		Q_hh[pair] += lr * minus_pi * (2 * real_qubits_dict[pair[0]] - 1)\
			* (2*real_qubits_dict[pair[1]] - 1) * gamma * reward

	for pair in Q_vh.keys():
		Q_vh[pair] += lr * minus_pi * old_state_encoding_tuple[pair[0]]\
			* (2*real_qubits_dict[pair[1]] - 1) * gamma * reward

	return Q_hh, Q_vh