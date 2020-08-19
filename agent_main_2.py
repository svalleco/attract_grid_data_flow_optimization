import pickle
import random
from pca_utils import get_normalized_values_per_week
from agent_utils import *
import multiprocessing as mp
import sys
from csv import reader
import matplotlib.pyplot as plt
import itertools
import os
from collections import namedtuple
from functools import reduce

'''
This file holds the training mechanism for the quantum agent. Multiple agents
run in parallel on their own CPU core.
'''

# The "State" object fields mean as follow:
#	modifiable_pc_decimal
#		It holds an iterable containing the principal components that the
#		agent is supposed to modify as float numbers.
#	modifiable_pc_qubits
#		It can be a list of lists of -1/1. Each sublist contains the qubit
#		representation of the component.
#	modifiable_pc_levels
#		It is a list of integer numbers representing the levels of the components.
State = namedtuple(\
	'State',\
	[\
		'modifiable_pc_decimal',\
		'modifiable_pc_qubits',\
		'modifiable_pc_levels',\
	]\
)

# The "Decimal_Binary_Pair" has 2 fields, one that holds float values and one
# that the -1/1 representation. It is used to facilitate easier access and
# clearer code for these to quantities for average read size, components etc.
Decimal_Binary_Pair = namedtuple(\
	'D_B_Pair',\
	[\
		'dec_rep',\
		'bin_rep',\
	]\
)

# The "Best_Action_Tuple" is used when the agent iterates through all the
# possible actions. It is used to hold quantities that will be needed in
# in the future to update the weights of the agent:
#	free_energy
#		Contains the free energy of a stat-action configuration as yielded
#		by the Quantum Boltzmann Machine.
#	action_integer_iterable
#		Contains the a level modification iterable per modifiable component
#	samples
#		Contains the configuration samples in 0/1.
#	used_visible_iterable
#		Contains the visibile qubit configuration.
Best_Action_Tuple = namedtuple(\
	'Best_Action_Tuple',\
	[\
		'free_energy',\
		'action_integer_iterable',\
		'samples',\
		'used_visible_iterable',\
	]\
)

def single_agent_train_1(index=0,pc_count=11, qbits_per_component=5,):
	'''
	It performs the training for a single agent

	index
		It is an index that is assigned to the current agent.

	pc_count
		It is the total number of components used in the training.

	qbits_per_component
		It is the number of qubits used per component/average read size.

	'''

	# "Q_hh" is a weights dict containing the weights between hidden units.
	Q_hh = dict()
	for i,ii in zip(tuple(range(4)),tuple(range(8,12))):
		for j,jj in zip(tuple(range(4,8)),tuple(range(12,16))):
			Q_hh[(i,j)] = 2 * random.random() - 1
			Q_hh[(ii,jj)] = 2 * random.random() - 1
	for i, j in zip(tuple(range(4,8)),tuple(range(12,16))):
		Q_hh[(i,j)] = 2 * random.random() - 1

	# "Q_vh" is a weights dict containing the weights between visible and hidden units.
	Q_vh = dict()
		# Fully connection between state and blue nodes
	for j in ( tuple(range(4)) + tuple(range(12,16)) ):
		for i in range( ( pc_count + 1 ) * qbits_per_component ):
			Q_vh[(i,j,)] = 2 * random.random() - 1
		# Fully connection between action and red nodes
	for j in ( tuple(range(4,8)) + tuple(range(8,12)) ):
		for i in range(\
				( pc_count + 1 ) * qbits_per_component,\
				( pc_count + 1 ) * qbits_per_component\
					+ action_qbits_count
			):
			Q_vh[(i,j,)] = 2 * random.random() - 1

	# File open for writing reward obtained for the current agent.
	reward_f = open(
		pre_path + agent_reward_history_folder +str(index)+'.csv','wt'
	)
	reward_f.write('general_index,reward_v,week_index\n' )

	general_index = week_ind = point_index = 0

	for _ in range(data_set_passes_count):

		# For each week
		for ars_iterable,\
			rc_iterable,\
			intial_pc_decimal_iterable in\
			zip(\
				ars_decimal_and_binary_per_week_iterable,\
				rc_decimal_and_binary_per_week_iterable,\
				map(\
					lambda e:\
						e[-1][:modifiable_princ_comp_count] ,\
					first_n_decimal_pc_per_week_iterable\
				),\
			):

			# Initialization for the first step for a week
			a_tuple = tuple( map( lambda e: create_ecoding( e , lvl_encodings_iterable ) , intial_pc_decimal_iterable ) )
			old_state_obj = State(\
				tuple(map(lambda a: a[3] , a_tuple)),\
				tuple(map(lambda a: a[2] , a_tuple)),\
				tuple(map(lambda a: a[4] , a_tuple)),\
			)
			del a_tuple

			#                       39
			for step_index in range(time_seq_length - 1,len(ars_iterable)):

				# step_index --> t + 1

				if point_index % 1000 == 0:
					print('run_index',index,': week =', week_ind, '; point =', point_index)

				point_index += 1

				# The agent acts in one of two ways. It either acts randomly or searches through
				# the action space for the action that yields the best free energy.
				if epsilon_greedy_coef == 0 or random.random() >= epsilon_greedy_coef:

					max_tuple = None

					# In the tested approach, the visible qubits iterable contains:
					#	-> 1 x average read size encoding
					#	-> 3 x principal component encodings from the agent
					#	-> 8 x principal component encodings from MonALISA
					#	-> 1 x encoding from the actions
					visible_without_action_tuple =\
						ars_iterable[ step_index - 1 ].bin_rep\
						+ tuple( itertools.chain.from_iterable(\
							old_state_obj.modifiable_pc_qubits ) )\
						+ rc_iterable[ step_index - 1 ].bin_rep

					# The agent iterates through all the available actions in the current state.
					for action_decimal_iterable, action_binary in get_possible_actions_on_state_func_2(\
							old_state_obj,\
							action_offsets_iterable,\
							modifiable_princ_comp_count,\
							max_pc_v,\
							act_to_enc_dict\
						):

						vis_iterable = visible_without_action_tuple + action_binary

						# The weights are transformed into a format that the DWAVE API understands.
						general_Q = create_general_Q_from(
							Q_hh,
							Q_vh,
							vis_iterable
						)

						# The samples are created.
						samples = list(SimulatedAnnealingSampler().sample_qubo(
							general_Q,
							num_reads=sample_count
						).samples())

						# random.shuffle(samples)

						# The free energy of the current state-action pair is evaluated.
						current_F = get_free_energy(
							get_3d_hamiltonian_average_value_1(
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

						# If the free energy of the current state-action pair is greater than the previous ones,
						# then it is memorized.
						if max_tuple is None or max_tuple.free_energy < current_F:
							max_tuple = Best_Action_Tuple(\
								current_F,\
								action_decimal_iterable,\
								samples,\
								vis_iterable,\
							)

				else:

					# An action is chosen at random. The same process is followed as in the previous branch.
					action_decimal_iterable, action_binary = random.choice(\
							get_possible_actions_on_state_func_2(\
								old_state_obj,\
								action_offsets_iterable,\
								modifiable_princ_comp_count,\
								max_pc_v,\
								act_to_enc_dict\
							)
						)

					vis_iterable = ars_iterable[ step_index - 1 ].bin_rep\
						+ tuple( itertools.chain.from_iterable(\
							old_state_obj.modifiable_pc_qubits ) )\
						+ rc_iterable[ step_index - 1 ].bin_rep\
						+ action_binary

					general_Q = create_general_Q_from(
						Q_hh,
						Q_vh,
						vis_iterable
					)

					samples = list(SimulatedAnnealingSampler().sample_qubo(
						general_Q,
						num_reads=sample_count
					).samples())

					# random.shuffle(samples)

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

					max_tuple = Best_Action_Tuple(\
						current_F,\
						action_decimal_iterable,\
						samples,\
						vis_iterable,\
					)

				# The new state is created based on the action the agent has
				# chosen.
				new_state_object = State(list(),list(),list())
				for lvl, act in zip(\
					old_state_obj.modifiable_pc_levels,\
					max_tuple.action_integer_iterable):

					new_state_object.modifiable_pc_levels.append( lvl + act )

					new_state_object.modifiable_pc_decimal.append(\
						lvl_encodings_iterable[\
							new_state_object.modifiable_pc_levels[-1]\
						][3]\
					)

					new_state_object.modifiable_pc_qubits.append(\
						lvl_encodings_iterable[\
							new_state_object.modifiable_pc_levels[-1]\
						][2]\
					)

				# The new components are written to shared memory array such that the prediction
				# process can infer the reward on it.
				for i, comp_v, i_act, act_v in\
					zip(\
						range(modifiable_princ_comp_count),\
						new_state_object.modifiable_pc_decimal,\
						range( index * modifiable_princ_comp_count , (index + 1) * modifiable_princ_comp_count ),\
						max_tuple.action_integer_iterable,\
					):

					shared_input_array_iterable[index][i] = comp_v

					shared_action_array[i_act] = act_v

				# The agent process waits for the inference process to be ready to infer.
				work_lock_iterable[index].acquire()

				# The agent process signals the inference process that it has put its components
				# in the shared array.
				before_inference_q.put( index )

				# Waits on the signal from the inference process.
				after_inference_lock_iterable[index].acquire()

				# The agent gets its reward.
				r = shared_output_array[index]

				# The agent writes its current reward to disk.
				reward_f.write(
					str(general_index)+','\
					+ str(r)+','\
					+ str(week_ind)+'\n'\
				)
				reward_f.flush()

				general_index += 1

				if step_index == time_seq_length - 1:
					old_samples = max_tuple.samples
					old_F = max_tuple.free_energy
					old_visible_iterable = max_tuple.used_visible_iterable
					old_r = r
				else:
					# The weights are updated based on the reward.
					Q_hh, Q_vh =\
						update_weights(
							Q_hh,
							Q_vh,
							old_samples,
							old_r,
							max_tuple.free_energy,
							old_F,
							old_visible_iterable,
							0.0001,
							0.8
						)
					old_samples = max_tuple.samples
					old_F = max_tuple.free_energy
					old_visible_iterable = max_tuple.used_visible_iterable
					old_r = r

			week_ind += 1

def poison_pill_inference_work():
	'''
	Represents what the inference part of the framework. The process waits to
	collect all the principal components from all the independent run agets at
	a step. Infers and distributes the results to the agents.
	'''
	from tensorflow import device
	from tensorflow.keras.models import load_model

	# The action file is for debuging. It logs the actions of the different agents.
	act_file = open( './agent_dumps/' + agent_log_file , 'wt' )

	act_file.write( 'step_index,' )

	for agent_ind in range(independent_run_count-1):

		for i_mod in range( modifiable_princ_comp_count ):

			act_file.write( 'agent_' + str(agent_ind) + '_action_' + str(i_mod) + ',' )

	for i_mod in range(modifiable_princ_comp_count-1):

		act_file.write( 'agent_' + str(independent_run_count-1) + '_action_' + str(i_mod) + ',' )

	act_file.write( 'agent_' + str(independent_run_count-1) + '_action_' + str(modifiable_princ_comp_count-1) + '\n' )

	step_index = 0

	# Iterates if multiple passes through the data set are desired.
	for _ in range(data_set_passes_count):

		# with device("/device:GPU:0"):
		if True:

			# The model for inference is loaded.
			local_model = load_model(k_model)

			# The array that is used to perform a sliding window approach.
			prediction_array = np.empty( ( independent_run_count , 40 , 12 ) )

			for i_week in range( len(ars_decimal_and_binary_per_week_iterable) ):

				# Initialize the working array.
				for a in range( time_seq_length - 1 ):
					prediction_array[ : , a + 1, 0 ]\
						= ars_decimal_and_binary_per_week_iterable[ i_week ][ a ].dec_rep

					prediction_array[ : , a + 1, 1 : ]\
						= first_n_decimal_pc_per_week_iterable[ i_week ][ a ]


				for i_step in range( time_seq_length - 1 , len( ars_decimal_and_binary_per_week_iterable[i_week] ) ):

					# i_step --> t + 1

					# Each independent run agent is notified that it can put its index in the shared queue.
					for i in range( independent_run_count ):
						work_lock_iterable[i].release()

					# All the components and average read sizes are moved one step back.
					prediction_array[ : , : time_seq_length - 1 , : ] =\
						prediction_array[ : , 1 : , : ]

					prediction_array[ : , time_seq_length - 1 , 0 ] =\
						ars_decimal_and_binary_per_week_iterable[ i_week ][ i_step ].dec_rep

					prediction_array[ : , time_seq_length - 1 , modifiable_princ_comp_count + 1 : ] =\
						rc_decimal_and_binary_per_week_iterable[ i_week ][ i_step ].dec_rep

					for i_run in range( independent_run_count ):

						run_index = before_inference_q.get()

						for a in range( modifiable_princ_comp_count ):
							prediction_array[ run_index , time_seq_length - 1 , a + 1 ] =\
								shared_input_array_iterable[ run_index ][ a ]

					act_file.write( str( step_index ) + ',' )

					for v in shared_action_array[:-1]:
						act_file.write( str(v) + ',' )

					act_file.write( str(shared_action_array[-1] ) + '\n' )

					act_file.flush()

					step_index += 1

					for v, i in zip(\
						local_model.predict(prediction_array),range(independent_run_count)):
						shared_output_array[i] = v[0]
						after_inference_lock_iterable[i].release()

def run_framework_main_1():
	'''
	It starts the multiple independent run testing algorithm.
	'''

	global\
		replica_count,\
		average_size,\
		sample_count,\
		time_seq_length,\
		ars_decimal_and_binary_per_week_iterable,\
		rc_decimal_and_binary_per_week_iterable,\
		first_n_decimal_pc_per_week_iterable,\
		work_lock_iterable,\
		before_inference_q,\
		modifiable_princ_comp_count,\
		shared_input_array_iterable,\
		after_inference_lock_iterable,\
		shared_output_array,\
		qb_per_pc_count,\
		max_pc_v,\
		act_to_enc_dict,\
		k_model,\
		action_qbits_count,\
		pre_path,\
		action_offsets_iterable,\
		independent_run_count,\
		lvl_encodings_iterable,\
		shared_action_array,\
		agent_reward_history_folder,\
		agent_log_file,\
		data_set_passes_count,\
		epsilon_greedy_coef

	'''
	big run is replica count 10 and average size 20

	2
		runs once through data set
		replica_count = 3
		average_size = 5

	3
		runs 3 times through data set
		replica_count = 3
		average_size = 5

	4
		runs once through data set
		replica_count = 3
		average_size = 5
		bigger reward range

	5
		runs once through data set
		replica_count = 3
		average_size = 5
		bigger lr 0.0001 -> 0.01

	6
		runs once through data set
		replica_count = 3
		average_size = 5
		epsilon greedy with epsilon = 0.3

	7
		runs once through data set
		replica_count = 3
		average_size = 5
		epsilon greedy with epsilon = 0.9

	8
		runs once through data set
		replica_count = 3
		average_size = 5
		epsilon greedy with epsilon = 0
		modified update_weights and get_hamilt
	'''

	epsilon_greedy_coef = 0
	data_set_passes_count = 1
	replica_count = 3
	average_size = 5
	sample_count = replica_count * average_size
	qb_per_pc_count = 5
	time_seq_length = 40
	action_offsets_iterable = (-1,0,1,)
	pre_path = '/export/home/proiecte/aux/mircea_marian.popa/'
	k_model = pre_path + 'agent_dumps/model_0643.hdf5'
	# pre_path = './'
	# k_model = 'pca_multiple_model_folders/models_320/model_0643.hdf5'
	agent_reward_history_folder = 'agent_rewards_history_8/'
	agent_log_file = 'actions_log_8.csv'
	modifiable_princ_comp_count = 3
	max_pc_v = 2**qb_per_pc_count
	independent_run_count = 38

	lvl_encodings_iterable = get_levels_encodings(qb_per_pc_count)

	# ars_decimal_and_binary_per_week_iterable
	#	Contains the float value and the qubit encoding for the average read size.
	# rc_decimal_and_binary_per_week_iterable
	#	Contains the "remaining components" aka the last 7 components.
	# first_n_decimal_pc_per_week_iterable
	#	Contains the first 39 compnents per week.
	ars_decimal_and_binary_per_week_iterable,\
		rc_decimal_and_binary_per_week_iterable,\
		first_n_decimal_pc_per_week_iterable =\
			list() , list() , list()

	for week_dict in pickle.load(open(pre_path+'agent_dumps/data_set_basedOn_6_july.p','rb')):

		ars_decimal_and_binary_per_week_iterable.append(
			tuple(
				map(
					lambda e:\
						Decimal_Binary_Pair(\
							e,\
							create_ecoding( e , lvl_encodings_iterable )[2],\
						),\
					week_dict['ars_list']
				)
			)
		)

		rc_decimal_and_binary_per_week_iterable.append(
			tuple(
				map(
					lambda comp_list:\
						Decimal_Binary_Pair(\
							comp_list,\
							tuple(\
								itertools.chain.from_iterable(\
									map(\
										lambda comp:\
											create_ecoding( comp , lvl_encodings_iterable )[2],\
										comp_list\
									)\
								)\
							)\
						),\
					week_dict['fixed_components_list']\
				)
			)
		)

		first_n_decimal_pc_per_week_iterable.append(
			week_dict['first_n_list']
		)

	work_lock_iterable = list()

	shared_input_array_iterable = list()

	after_inference_lock_iterable = list()

	for _ in range(independent_run_count):

		work_lock_iterable.append( mp.Lock() )
		work_lock_iterable[-1].acquire()

		shared_input_array_iterable.append( mp.RawArray( 'f' , modifiable_princ_comp_count ) )

		after_inference_lock_iterable.append( mp.Lock() )
		after_inference_lock_iterable[-1].acquire()

	before_inference_q = mp.Queue()

	shared_output_array = mp.RawArray( 'f' , independent_run_count )

	shared_action_array = mp.RawArray( 'i' , independent_run_count * modifiable_princ_comp_count)

	def create_all_possible_actions_encodings(acc_tuple, partition_index):
		a = list()
		if partition_index == modifiable_princ_comp_count:
			for act in action_offsets_iterable:
				a.append( acc_tuple + (act,) )
			return a
		for act in action_offsets_iterable:
			a += create_all_possible_actions_encodings( acc_tuple + (act,) , partition_index + 1 )
		return a

	action_qbits_count = 2
	while 2**action_qbits_count < len(action_offsets_iterable)**modifiable_princ_comp_count:
		action_qbits_count += 1

	act_to_enc_dict = dict()
	i = 0
	for act in create_all_possible_actions_encodings( tuple() , 1 ):
		act_to_enc_dict[act] = from_number_to_qubits( i , action_qbits_count )
		i += 1

	process_list = [ mp.Process(target=poison_pill_inference_work) , ]
	process_list[0].start()

	for i in range(independent_run_count):
		process_list.append(\
			mp.Process(target=single_agent_train_1, args=(i,))
		)
		process_list[-1].start()

	for p in process_list: p.join()

if __name__ == '__main__':
	run_framework_main_1()