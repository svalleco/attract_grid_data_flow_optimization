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
'''
#!/home/mircea/.pyenv/versions/3.6.1/bin/python3.6
'''

def set_up_first_n_sequences_and_read_size(n=39):
	if False:
		a_tuple = get_normalized_values_per_week( 'tm_tren_ars_pc_per_week_10_may.p' )
		pickle.dump(
			a_tuple,
			open('./pca_dumps/norm_ars_pc_tren_per_week_10_may.p','wb')
		)
		exit(0)
	if True:
		a_list = pickle.load(open('./pca_dumps/norm_ars_pc_tren_per_week_10_may.p','rb'))

	def set_up_first_n_and_ars(week_list):
		first_n_list = list()
		for b_list in week_list[:n]:
			first_n_list.append(b_list[1:-1])
		return\
			(
				first_n_list,
				tuple(map(lambda e: e[0],week_list))
			)

	print(len(a_list[0][0]))

	a_list = tuple(map(lambda w_l: set_up_first_n_and_ars(w_l),a_list))

	pickle.dump(
		a_list,
		open('./agent_dumps/data_set_basedOn_10_may.p','wb')
	)

def set_up_first_n_components__remaining_components__read_size(modifiable_components_count=3,n=39):
	a_list = list()
	for week_list in pickle.load(open('./pca_dumps/norm_ars_pc_tren_per_week_10_may.p','rb')):
		a_list.append(\
			{\
				'first_n_list':list(),\
				'ars_list':tuple(map(lambda e: e[0],week_list)),\
				'fixed_components_list':tuple(map(lambda e: e[modifiable_components_count + 1:-1], week_list)),\
			}\
		)
		for b_list in week_list[:n]:
			a_list[-1]['first_n_list'].append(b_list[1:-1])

	pickle.dump(
		a_list,
		open('./agent_dumps/data_set_basedOn_6_july.p','wb')
	)

def single_agent_train(index=0,pc_count=11, qbits_per_component=5, lvl_jumping_actions_count=3, actions_count=11):

	if lvl_jumping_actions_count == 1 or lvl_jumping_actions_count % 2 == 0:
		print('Please specify an even number of actions that is greater than 1 !')
		return

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

	def agent_step_1(current_state_tuple):
		max_tuple = None

		for act in get_possible_actions_on_state_func_0(\
				current_state_tuple[qb_per_pc_count:], action_offsets_iterable,\
				small_part_len,big_partition_start,small_part_count,\
				small_limit,big_limit,act_to_enc_dict):

			vis_iterable = current_state_tuple + act

			if False:
				print('in agent step : vis_iterable :', vis_iterable)
				print('in agent step : len(vis_iterable) :', len(vis_iterable))
				# exit(0)

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

			if max_tuple is None or max_tuple[0] < current_F:
				max_tuple = ( current_F , act , samples , vis_iterable )

		return\
			(\
				apply_action_func(\
					act,\
					current_state_tuple[qb_per_pc_count:],\
					small_part_len,\
					small_part_count,\
					big_partition_start,\
					enc_to_act_dict,\
				),\
				max_tuple[0],     # F value
				max_tuple[2],     # samples
				max_tuple[3],     # used iterable vector
				max_tuple[1],     # action index
				current_state_tuple,    # old_position
			)

	reward_f = open(
		'./agent_rewards_history/'+str(index)+'.csv','wt'
	)
	reward_f.write( 'general_index,reward_v,week_index\n' )

	general_index = 0

	for initial_pc_list, initial_ars_list, ars_list,week_ind in zip(\
			initial_qbit_pc_state_per_week_list,\
			initial_qbit_ars_per_week_list,\
			ars_per_week_list,\
			range(len(ars_per_week_list)),\
		):

		if False:
			print('average read size',initial_ars_list)
			print('iniatial pc',initial_pc_list)
			exit(0)

		agent_pc_tuple, current_F, current_samples,\
			current_vis_iterable, action_index, old_position_tuple = agent_step_1(tuple(initial_ars_list) + tuple(initial_pc_list))

		for ars_index in range(40,len(ars_list)):

			agent_pc_tuple, future_F, future_samples, future_vis_iterable,\
				action_index, old_position_tuple =\
				agent_step_1(\
					tuple(agent_pc_tuple) + ars_list[ars_index]
				)

			r = get_reward_through_model(
						agent_pc_tuple,
						ars_list[ars_index],
						k_model,
						qb_per_pc_count,
						lvl_encodings_iterable,
						train_lock
					)

			reward_f.write(
				str(general_index)+','\
				+ str(r)+','\
				+ str(week_ind)+'\n'\
			)
			reward_f.flush()

			general_index += 1

			Q_hh, Q_vh =\
				update_weights(
					Q_hh,
					Q_vh,
					current_samples,
					r,
					future_F,
					current_F,
					current_vis_iterable,
					0.0001,
					0.8
				)

			# print(index,'Finished a lap !')

			current_F, current_samples, current_vis_iterable =\
				future_F, future_samples, future_vis_iterable

def run_framework_main():
	global replica_count, average_size, sample_count,\
		initial_qbit_pc_state_per_week_list, initial_qbit_ars_per_week_list,\
		ars_per_week_list,\
		action_offsets_iterable, small_part_len,\
		big_partition_start, small_part_count,\
		small_limit, big_limit, act_to_enc_dict,\
		enc_to_act_dict,action_qbits_count, k_model,\
		lvl_encodings_iterable, partitions_total_count,\
		qb_per_pc_count, train_lock

	replica_count = 10
	average_size = 50
	sample_count = replica_count * average_size
	qb_per_pc_count = 5
	action_offsets_iterable = (-1,0,1)
	partitions_total_count = 3
	k_model = keras.models.load_model('./pca_multiple_model_folders/models_320/model_0643.hdf5')
	# k_model = './pca_multiple_model_folders/models_320/model_0643.hdf5'

	lvl_encodings_iterable = get_levels_encodings(qb_per_pc_count)
	initial_qbit_pc_state_per_week_list = list()
	initial_qbit_ars_per_week_list = list()
	ars_per_week_list = list()
	for first_n_pc_list, ars_list in pickle.load(open('./agent_dumps/data_set_basedOn_10_may.p','rb')):
		initial_qbit_ars_per_week_list.append(
			create_ecoding(
				ars_list[len(first_n_pc_list)-1],
				lvl_encodings_iterable
			)[2]
		)
		initial_qbit_pc_state_per_week_list.append(list())
		for pc_value in first_n_pc_list[-1]:
			initial_qbit_pc_state_per_week_list[-1] += create_ecoding( pc_value , lvl_encodings_iterable )[2]
		ars_per_week_list.append(
			list(
				map(
					lambda e: create_ecoding( e , lvl_encodings_iterable )[2],
					ars_list
				)
			)
		)

	small_part_len = len(initial_qbit_pc_state_per_week_list[0]) // partitions_total_count
	small_part_count = partitions_total_count - 1
	big_partition_start = small_part_len * small_part_count
	small_limit = 2**small_part_len - 1
	big_limit = 2**(small_part_len + len(initial_qbit_pc_state_per_week_list[0]) % partitions_total_count) - 1

	def create_all_possible_actions_encodings(acc_tuple, partition_index):
		a = list()
		if partition_index == partitions_total_count:
			for act in action_offsets_iterable:
				a.append( acc_tuple + (act,) )
			return a
		for act in action_offsets_iterable:
			a += create_all_possible_actions_encodings( acc_tuple + (act,) , partition_index + 1 )
		return a

	action_qbits_count = 2
	while 2**action_qbits_count < len(action_offsets_iterable)**partitions_total_count:
		action_qbits_count += 1

	act_to_enc_dict = dict()
	enc_to_act_dict = dict()
	i = 0
	for act in create_all_possible_actions_encodings( tuple() , 1 ):
		act_to_enc_dict[act] = from_number_to_qubits( i , action_qbits_count )
		enc_to_act_dict[ act_to_enc_dict[act] ] = act
		i += 1

	if False:
		print('all posib actions',action_qbits_count)
		print('all posib actions',tuple(act_to_enc_dict.items()))
		exit(0)

	train_lock = Lock()

	# Pool(1).map(single_agent_train, range(20))

	for i in range(10):
		single_agent_train(i)

def single_agent_train_1(index=0,pc_count=11, qbits_per_component=5, lvl_jumping_actions_count=3, actions_count=11):

	if lvl_jumping_actions_count == 1 or lvl_jumping_actions_count % 2 == 0:
		print('Please specify an even number of actions that is greater than 1 !')
		return

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
		for i in range( ( pc_count + 1 ) * qbits_per_component ):
			Q_vh[(i,j,)] = 2 * random.random() - 1
		# Fully connection between action and red nodes
	for j in ( tuple(range(4,8)) + tuple(range(8,12)) ):
		for i in range(\
				( pc_count + 1 ) * qbits_per_component,\
				( pc_count + 1 ) * qbits_per_component + action_qbits_count,\
			):
			Q_vh[(i,j,)] = 2 * random.random() - 1

	def agent_step_1(current_state_tuple, new_remainig_components):
		max_tuple = None

		for act in get_possible_actions_on_state_func_1(\
				current_state_tuple[qb_per_pc_count:], action_offsets_iterable,\
				qb_per_pc_count,princ_comp_count,max_v,act_to_enc_dict):

			vis_iterable = current_state_tuple + act

			if False:
				print('in agent step : vis_iterable :', vis_iterable)
				print('in agent step : len(vis_iterable) :', len(vis_iterable))
				# exit(0)

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

			if max_tuple is None or max_tuple[0] < current_F:
				max_tuple = ( current_F , act , samples , vis_iterable )

		return\
			(\
				apply_action_func_1(\
					act,\
					current_state_tuple[qb_per_pc_count:],\
					qb_per_pc_count,\
					princ_comp_count,\
					enc_to_act_dict,\
					new_remainig_components,\
				),\
				max_tuple[0],     		# F value
				max_tuple[2],     		# samples
				max_tuple[3],     		# used iterable vector
				max_tuple[1],     		# action index
				current_state_tuple,    # old_position
			)

	reward_f = open(
		pre_path + 'agent_rewards_history/'+str(index)+'.csv','wt'
	)
	reward_f.write('general_index,reward_v,week_index\n' )

	# from tensorflow.keras.models import load_model
	# local_model = load_model(k_model)

	general_index = week_ind = point_index = 0

	for initial_pc_list, initial_ars_list, ars_list,week_ind,\
		remaining_components in zip(\
			initial_qbit_pc_state_per_week_list,\
			initial_qbit_ars_per_week_list,\
			ars_per_week_list,\
			range(len(ars_per_week_list)),\
			remaining_components_week_list,\
		):

		agent_pc_tuple, current_F, current_samples,\
			current_vis_iterable, action_index, old_position_tuple = agent_step_1(\
			tuple(initial_ars_list) + tuple(initial_pc_list),\
			remaining_components[39],\
		)

		for ars_index in range(40,len(ars_list)):

			if point_index % 1000 == 0:
			# if True:
				print('run_index',index,': week =', week_ind, '; point =', point_index)

			point_index += 1

			agent_pc_tuple, future_F, future_samples, future_vis_iterable,\
				action_index, old_position_tuple =\
				agent_step_1(\
					tuple(agent_pc_tuple) + ars_list[ars_index],
					remaining_components[ars_index],
				)

			if False:
				r = get_reward_through_model(\
					agent_pc_tuple,\
					ars_list[ars_index],\
					local_model,\
					qb_per_pc_count,\
					lvl_encodings_iterable,\
					train_lock\
				)
			if True:
				if index == 0:
					poison_q.put( False )

				# print( index , 'Will try to get reward !' )

				r = get_reward_through_model_2(\
					agent_pc_tuple,\
					ars_list[ars_index],\
					qb_per_pc_count,\
					lvl_encodings_iterable,\
					index,\
					shared_input_array_iterable[index],\
					shared_output_array,\
					before_inference_q,\
					after_inference_lock_iterable[index],\
					work_lock_iterable[index],\
				)

			reward_f.write(
				str(general_index)+','\
				+ str(r)+','\
				+ str(week_ind)+'\n'\
			)
			reward_f.flush()

			general_index += 1

			Q_hh, Q_vh =\
				update_weights(
					Q_hh,
					Q_vh,
					current_samples,
					400 * r,
					future_F,
					current_F,
					current_vis_iterable,
					0.0001,
					0.8
				)

			# print(index,'Finished a lap !')

			current_F, current_samples, current_vis_iterable =\
				future_F, future_samples, future_vis_iterable

			# print('Finished iteration !')

		week_ind += 1

	if index == 0:
		poison_q.put( True )

def poison_pill_inference_work():
	prediction_array = np.empty( ( independent_run_count , 40 , 12 ) )

	from tensorflow import device
	from tensorflow.keras.models import load_model

	with device("/device:GPU:0"):

		local_model = load_model(k_model)

		while True:

			finshed_work_flag = poison_q.get()

			if finshed_work_flag:
				break

			for i in range( independent_run_count ):
				work_lock_iterable[i].release()

			for _ in range( independent_run_count ):
				run_index = before_inference_q.get()
				# print('predictor Got info from',run_index)
				for i,j in zip(range( 40 ),range( 0 , 480 , 12 )):
					for l,k in zip( range( 12 ) , range( j , j + 12 ) ):
						prediction_array[run_index,i,l] =\
							shared_input_array_iterable[run_index][k]

			for v, i in zip(\
				local_model.predict(prediction_array),range(independent_run_count)):
				shared_output_array[i] = v[0]
				after_inference_lock_iterable[i].release()

def run_framework_main_1():
	global replica_count, average_size, sample_count,\
		initial_qbit_pc_state_per_week_list, initial_qbit_ars_per_week_list,\
		ars_per_week_list,after_inference_lock_iterable,\
		action_offsets_iterable, independent_run_count,\
		small_limit, big_limit, act_to_enc_dict,\
		enc_to_act_dict,action_qbits_count, k_model,\
		lvl_encodings_iterable, partitions_total_count,\
		qb_per_pc_count, train_lock, remaining_components_week_list,\
		max_v, princ_comp_count, before_inference_q,\
		shared_input_array_iterable, shared_output_array,\
		poison_q, work_lock_iterable,pre_path

	replica_count = 8
	average_size = 12
	sample_count = replica_count * average_size
	qb_per_pc_count = 5
	action_offsets_iterable = (-1,0,1)
	# k_model = keras.models.load_model('./pca_multiple_model_folders/models_320/model_0643.hdf5')
	pre_path = '/export/home/proiecte/aux/mircea_marian.popa/'
	if False:
		k_model = pre_path + 'pca_multiple_model_folders/models_320/model_0643.hdf5'
	if True:
		k_model = pre_path + 'agent_dumps/model_0643.hdf5'
	princ_comp_count = 3
	max_v = 2**qb_per_pc_count
	independent_run_count = 38

	lvl_encodings_iterable = get_levels_encodings(qb_per_pc_count)
	initial_qbit_pc_state_per_week_list = list()
	initial_qbit_ars_per_week_list = list()
	ars_per_week_list = list()
	remaining_components_week_list = list()
	for week_dict in pickle.load(open(pre_path+'agent_dumps/data_set_basedOn_6_july.p','rb')):
		initial_qbit_ars_per_week_list.append(
			create_ecoding(
				week_dict['ars_list'][len(week_dict['first_n_list'])-1],
				lvl_encodings_iterable
			)[2]
		)
		remaining_components_week_list.append(list())
		for ex in week_dict['fixed_components_list']:
			remaining_components_week_list[-1].append( tuple() )
			for el in ex:
				remaining_components_week_list[-1][-1] += create_ecoding( el , lvl_encodings_iterable )[2]

		initial_qbit_pc_state_per_week_list.append(list())
		for pc_value in week_dict['first_n_list'][-1]:
			initial_qbit_pc_state_per_week_list[-1] += create_ecoding( pc_value , lvl_encodings_iterable )[2]
		ars_per_week_list.append(
			list(
				map(
					lambda e: create_ecoding( e , lvl_encodings_iterable )[2],
					week_dict['ars_list']
				)
			)
		)

	# print('Finished setting up stuff !')

	def create_all_possible_actions_encodings(acc_tuple, partition_index):
		a = list()
		if partition_index == princ_comp_count:
			for act in action_offsets_iterable:
				a.append( acc_tuple + (act,) )
			return a
		for act in action_offsets_iterable:
			a += create_all_possible_actions_encodings( acc_tuple + (act,) , partition_index + 1 )
		return a

	action_qbits_count = 2
	while 2**action_qbits_count < len(action_offsets_iterable)**princ_comp_count:
		action_qbits_count += 1

	act_to_enc_dict = dict()
	enc_to_act_dict = dict()
	i = 0
	for act in create_all_possible_actions_encodings( tuple() , 0 ):
		act_to_enc_dict[act] = from_number_to_qubits( i , action_qbits_count )
		enc_to_act_dict[ act_to_enc_dict[act] ] = act
		i += 1

	train_lock = mp.Lock()

	# print('Will start single_agent_train_1 !')

	poison_q = mp.Queue()
	before_inference_q = mp.Queue()
	after_inference_lock_iterable = tuple()
	work_lock_iterable = tuple()
	shared_input_array_iterable = tuple()
	for _ in range( independent_run_count ):
		after_inference_lock_iterable += ( mp.Lock() , )
		after_inference_lock_iterable[-1].acquire()
		work_lock_iterable += ( mp.Lock() , )
		work_lock_iterable[-1].acquire()
		shared_input_array_iterable += ( mp.RawArray( 'f' , 12 * 40 ) , )
	shared_output_array = mp.RawArray( 'f' , independent_run_count )

	process_list = [ mp.Process(target=poison_pill_inference_work) , ]
	process_list[0].start()

	for i in range(independent_run_count):
		process_list.append(\
			mp.Process(target=single_agent_train_1, args=(i,))
		)
		process_list[-1].start()

	for p in process_list: p.join()

	# mp.Pool(independent_run_count).map(single_agent_train_1, range(independent_run_count))
	# for i in range(1):
	# 	single_agent_train_1(i)

def plot_results(index):
	a_list = pickle.load(open('pipe_'+str(index)+'.p','rb'))

	def aux_get_list(ind):
		b_list =\
			tuple(\
				map(\
					lambda week_list:\
						tuple(\
							map(\
								lambda e: e[ind], week_list\
							)\
						),\
					pickle.load( open( './pca_dumps/tm_tren_ars_pc_per_week_10_may.p' , 'rb' ) )\
				)\
			)

		min_v, max_v = min(map(lambda week_list: min(week_list), b_list)),\
			max(map(lambda week_list: max(week_list), b_list))

		return\
			tuple(\
				itertools.chain.from_iterable(\
					map(\
						lambda week_list:\
							map( lambda e: ( e - min_v ) / ( max_v - min_v ) , week_list[ 39 : ] ),\
						b_list,\
					)\
				)\
			)

	b_list, c_list = aux_get_list(1), aux_get_list(2)

	# print(\
	# 	'Points better or equal to the original:',
	# 	sum( map( lambda p: 1 if p[0] >= p[1] else 0 , zip( a_list , b_list ) ) ),
	# 	'/',len(b_list)
	# )

	plt.plot(
		range(len(a_list)),
		a_list,
		label='Agent Trend'
	)
	plt.plot(
		range(len(a_list)),
		a_list,
		'ro',
		label='Agent Trend'
	)


	plt.plot(
		range(len(b_list)),
		b_list,
		label='MonALISA Trend'
	)
	plt.plot(
		range(len(b_list)),
		b_list,
		'go',
		label='MonALISA Trend'
	)

	# plt.plot(
	# 	range(len(c_list)),
	# 	c_list,
	# 	'ko',
	# 	label='A.R.S.',
	# )

	plt.xlabel('Temporarily Ordered Index')
	plt.ylabel('Trend')
	plt.legend()

	plt.show()

def dump_results(index):
	rewards_folder = './agent_rewards_history_'+str(index)+'/'

	divisor_n = 0

	a_list = list()

	for csv_file_reader in map(\
			lambda pair: reader( open( pair[0] , 'rt' ) ),\
			filter(\
				lambda pair: pair[1] > 200,\
				map(
					lambda fn:\
						(\
							rewards_folder + fn,\
							len(\
								open( rewards_folder + fn , 'rt' ).read().split('\n'),\
							),\
						),
					os.listdir( rewards_folder )
				)
			)
		):

		next(csv_file_reader)

		for ind,line_list in enumerate(csv_file_reader):
			if len(line_list) > 0:
				if ind == len(a_list):
					a_list.append( [ float(line_list[1]) , 1 ] )
				else:
					a_list[ind][0] += float(line_list[1])
					a_list[ind][1] += 1

		divisor_n += 1

	a_list = tuple(
		map(
			lambda e: e[0] / divisor_n,
			filter(
				lambda p: p[1] == divisor_n,
				a_list
			)
		)
	)

	pickle.dump(a_list, open('pipe_'+str(index)+'.p','wb'))

def dump_results_1():
	previous_a_list = pickle.load(open('pipe.p','rb'))

	a_list = list()

	divisor_n = 0

	for opened_file_content_gen in map(\
			lambda pair:\
				map(\
					lambda line: float( line.split(',')[1]  ),
					filter(\
						lambda line:\
							2 <= len(line) < 50\
							and 'general_index' not in line\
							and len(previous_a_list) <= float( line.split(',')[0] ),\
						open( pair[0] , 'rt' ).read().split('\n')\
					),\
				),\
			filter(\
				lambda pair: pair[1] > 200,\
				map(
					lambda fn:\
						(\
							'./agent_rewards_history/' + fn,\
							len(\
								open( './agent_rewards_history/' + fn , 'rt' ).read().split('\n'),\
							),\
						),
					os.listdir( './agent_rewards_history/' )
				)
			)
		):

		for ind,line_list in enumerate(opened_file_content_gen):
			if ind == len(a_list):
				a_list.append( [ line_list , 1 ] )
			else:
				a_list[ind][0] += line_list
				a_list[ind][1] += 1

		divisor_n += 1

	a_list = tuple(
		map(
			lambda e: e[0] / divisor_n,
			filter(
				lambda p: p[1] == divisor_n,
				a_list
			)
		)
	)

	pickle.dump(previous_a_list + a_list, open('pipe_1.p','wb'))

if __name__ == '__main__':
	# run_framework_main()
	# set_up_first_n_sequences_and_read_size()
	# a = get_levels_encodings(5)
	# for x in a: print(x)
	# print()
	# print(-1,create_ecoding(-1, a))

	# print(from_qubits_to_number((-1,1,-1)))
	# for i in range(8):
	# 	print(i,from_number_to_qubits(i, 3))

	# run_framework_main_1()
	# set_up_first_n_components__remaining_components__read_size()
	plot_results(int(sys.argv[1]))
	# dump_results_1()
	# dump_results( int(sys.argv[1]) )

	# a_list = pickle.load(open('pipe.p','rb'))
	# plt.plot( range(len(a_list)) , a_list )
	# plt.show()

	'''
	pca_cumsum used:
	[0.86d51914  0.92107248 0.94547703 0.95990777 0.96971694 0.97624362
 		0.97978872 0.98254824 0.98486565 0.98678833 0.98845562]
	'''
