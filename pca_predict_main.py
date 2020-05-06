import os
import matplotlib.pyplot as plt
from shutil import copyfile
import pickle
import itertools
from sklearn.decomposition import PCA
import numpy as np
from multiprocessing import Pool, Queue
import json
import csv
import random
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf
from collections import namedtuple
import itertools
from pca_utils import *

MATRIXES_FOLDER_LIST = (\
	'./remote_host_0/log_folder/',\
	'./remote_host_1/log_folder/',\
	'./remote_host_1/log_folder/',\
)

WEEK_TIME_MOMENTS = (\
	(1579215600000, 1579875041000),\
	(1580511600000, 1581289199000),\
	(1581289200000, 1581980399000),\
)

UNANSWERED_PATH_TUPLE = (\
	'/data/mipopa/unanswered_query_dump_folder_0/',\
	'/optane/mipopa/unanswered_query_dump_folder_1/',\
	'/optane/mipopa/unanswered_query_dump_folder_2/',\
)

PCA_DUMP_FOLDER='./pca_dumps/'

RAW_QUERY_FILES = (\
	'/data/mipopa/apicommands.log',
	'/data/mipopa/apicommands2.log',
	'/data/mipopa/apicommands3.log',
)

def create_folder_and_copy_files():
	folder_path = '/home/mircea/cern_log_folder/remote_host/log_folder/'

	os.mkdir('./remote_host_0/')

	os.mkdir('./remote_host_0/log_folder/')

	for fn in os.listdir(folder_path):
		if WEEK_TIME_MOMENTS[0][0] <= int(fn.split('_')[0]) <= WEEK_TIME_MOMENTS[0][1]:
			copyfile(
				folder_path + fn,
				'./remote_host_0/log_folder/' + fn,
			)

def get_min_max_time_moments_per_matrixes(list_of_spool_folders):
	for folder_path in list_of_spool_folders:
		print(folder_path + '   :   ' + min(os.listdir(folder_path)) + '   ' + max(os.listdir(folder_path)))

def compare_gaps_in_time_matrix_vs_thp():
	get_dem_moments = lambda folder_name, tm_tuple: sorted(
		filter(
			lambda e: tm_tuple[0] <= e <= tm_tuple[1],
			map(
				lambda fn: int(fn.split('_')[0]),
				filter(
					lambda fn: 'demotion' in fn,
					os.listdir(folder_name)
				)
			)
		)
	)

	get_differences_in_seconds = lambda l: list(
		map(
			lambda p: p[1] - p[0],
			zip(
				l[:-1],
				l[1:]
			)
		)
	)

	to_dump = (
		get_differences_in_seconds(get_dem_moments(
			MATRIXES_FOLDER_LIST[0],WEEK_TIME_MOMENTS[0])),
		get_differences_in_seconds(get_dem_moments(
			MATRIXES_FOLDER_LIST[1],WEEK_TIME_MOMENTS[1])),
		get_differences_in_seconds(get_dem_moments(
			MATRIXES_FOLDER_LIST[1],WEEK_TIME_MOMENTS[2])),
	)

	print_mean=lambda index: print('average distance for week '\
		+ str(index) + ': ' + str(sum(to_dump[index])/len(to_dump[index])))

	print_mean(0)
	print_mean(1)
	print_mean(2)

	pickle.dump(
		to_dump,
		open(
			'pipe.p',
			'wb'
		)
	)

def plot_differences():
	a,b,c = pickle.load(
		open(
			'pipe.p',
			'rb'
		)
	)

	min_v, max_v=\
		min(a + b + c), max(a + b + c)

	diff_dict = dict()
	for d in itertools.chain.from_iterable((a,b,c)):
		if d in diff_dict:
			diff_dict[d] += 1
		else:
			diff_dict[d] = 1

	print(
		sorted( diff_dict.items() , key=lambda p: p[1] , reverse=True)[:10]
	)

	plt.plot(
		range(len(a)),
		a,
		'bo'
	)
	plt.plot(
		(len(a)-1,len(a)-1,),
		(min_v, max_v)
	)
	plt.plot(
		list(map(lambda e: e + len(a),range(len(b)))),
		b,
		'ro'
	)
	plt.plot(
		(len(a) + len(b) - 1, len(a)+len(b) - 1,),
		(min_v, max_v)
	)
	plt.plot(
		list(map(lambda e: e + len(b) + len(a),range(len(c)))),
		c,
		'go'
	)
	plt.show()

	'''
	average distance for week 0: 77434.14470842332
	average distance for week 1: 70901.74727987565
	average distance for week 2: 97927.02018479034
	'''

def dump_whole_matrices():
	'''
	This does 3 things in the following order:
		-> Reads the distance and demotion matrices from raw files and parses them into dictionarie
		-> Generates a minimal viable set of clients and storage elements.
		-> Generates the set of whole matrices.
	'''
	def get_distance_matrix_per_week(time_interval, matrices_folder):
		distance_list = list()
		for fn in os.listdir(matrices_folder):
			if 'distance' in fn:
				tm = int(fn.split('_')[0])
				if time_interval[0] <= tm <= time_interval[1]:
					distance_list.append((
						tm,
						get_dist_dict_by_time(tm, matrices_folder)
						)
					)
		return sorted(distance_list)

	def get_demotion_matrix_per_week(time_interval, matrices_folder):
		demotion_list = list()
		for fn in os.listdir(matrices_folder):
			if 'demotion' in fn:
				tm = int(fn.split('_')[0])
				if time_interval[0] <= tm <= time_interval[1]:
					demotion_list.append((
						tm,
						get_dem_dict_by_time(tm, matrices_folder))
					)
		return sorted(demotion_list)

	def get_clients_and_ses_minimal_list(distance_lists_list, demotion_lists_list):
		clients_sorted_list = sorted( distance_lists_list[0][0][1].keys() )
		se_sorted_list = sorted(demotion_lists_list[0][0][1].keys())

		for distance_list in distance_lists_list:
			for _ , cl_dict in distance_list:

				cl_to_remove_list = list()
				for cl in clients_sorted_list:
					if cl not in cl_dict:
						cl_to_remove_list.append(cl)
				for cl in cl_to_remove_list: clients_sorted_list.remove(cl)

				for se_dict in cl_dict.values():
					se_to_remove_list = list()
					for se in se_sorted_list:
						if se not in se_dict:
							se_to_remove_list.append(se)
					for se in se_to_remove_list: se_sorted_list.remove(se)

		for demotion_list in demotion_lists_list:
			for _, se_dict in demotion_list:
				se_to_remove_list = list()
				for se in se_sorted_list:
					if se not in se_dict:
						se_to_remove_list.append(se)
				for se in se_to_remove_list: se_sorted_list.remove(se)

		return clients_sorted_list, se_sorted_list

	def get_complete_distance_matrix(distance_list, demotion_list, clients_sorted_list, se_sorted_list):

		print( 'clients numbers: ' + str(set( map( lambda p: len(p[1].keys()) , distance_list ) )))

		print('se numbers 0: '\
			+ str(set( map( lambda p: len(p[1][tuple(p[1].keys())[0]].keys()) , distance_list ) )))

		print('se numbers 1: ' + str(set( map( lambda p: len(p[1].keys()) , demotion_list ) )))

		dist_i = 0

		while distance_list[dist_i][0] < demotion_list[0][0]: dist_i += 1

		whole_distance_list = list()

		while dist_i < len(distance_list):

			dem_i = 0
			while demotion_list[dem_i+1][0] <= distance_list[dist_i][0]: dem_i += 1

			res_list = list()

			for cl in clients_sorted_list:
				for se in se_sorted_list:
					# if se != ('spbsu', 'se'):
					res_list.append(
						distance_list[dist_i][1][cl][se]\
						+ demotion_list[dem_i][1][se]
					)

			whole_distance_list.append((
				distance_list[dist_i][0],
				res_list
			))

			dist_i += 1

		return whole_distance_list

	def get_complete_distance_matrix_1(distance_list, demotion_list, clients_sorted_list, se_sorted_list):

		print( 'clients numbers: ' + str(set( map( lambda p: len(p[1].keys()) , distance_list ) )))

		print('se numbers 0: '\
			+ str(set( map( lambda p: len(p[1][tuple(p[1].keys())[0]].keys()) , distance_list ) )))

		print('se numbers 1: ' + str(set( map( lambda p: len(p[1].keys()) , demotion_list ) )))

		dist_i = 0
		dem_i = 0

		while distance_list[dist_i][0] < demotion_list[0][0]: dist_i += 1

		while demotion_list[dem_i][0] < distance_list[0][0]: dem_i += 1

		whole_distance_list = list()

		tm_already_added_set = set()

		aux_ind = 0
		while dist_i < len(distance_list):

			while aux_ind < len(demotion_list) - 1\
				and demotion_list[aux_ind + 1][0] <= distance_list[dist_i][0] : aux_ind += 1

			res_list = list()

			for cl in clients_sorted_list:
				for se in se_sorted_list:
					res_list.append(
						distance_list[dist_i][1][cl][se]\
						+ demotion_list[aux_ind][1][se]
					)

			tm_already_added_set.add( distance_list[dist_i][0] )

			whole_distance_list.append(
				(
					distance_list[dist_i][0],
					res_list,
				)
			)

			dist_i += 1

		aux_ind = 0
		while dem_i < len(demotion_list):

			while aux_ind < len(distance_list) - 1\
				and distance_list[aux_ind + 1][0] <= demotion_list[dem_i][0] : aux_ind += 1

			if demotion_list[dem_i][0] not in tm_already_added_set:

				res_list = list()

				for cl in clients_sorted_list:
					for se in se_sorted_list:
						res_list.append(
							distance_list[aux_ind][1][cl][se]\
							+ demotion_list[dem_i][1][se]
						)

				whole_distance_list.append(
					(
						demotion_list[dem_i][0],
						res_list,
					)
				)

			dem_i += 1

		return sorted(whole_distance_list)

	distance_lists_list = (
		get_distance_matrix_per_week(WEEK_TIME_MOMENTS[0],MATRIXES_FOLDER_LIST[0]),
		get_distance_matrix_per_week(WEEK_TIME_MOMENTS[1],MATRIXES_FOLDER_LIST[1]),
		get_distance_matrix_per_week(WEEK_TIME_MOMENTS[2],MATRIXES_FOLDER_LIST[1]),
	)

	demotion_lists_list = (
		get_demotion_matrix_per_week(WEEK_TIME_MOMENTS[0],MATRIXES_FOLDER_LIST[0]),
		get_demotion_matrix_per_week(WEEK_TIME_MOMENTS[1],MATRIXES_FOLDER_LIST[1]),
		get_demotion_matrix_per_week(WEEK_TIME_MOMENTS[2],MATRIXES_FOLDER_LIST[1]),
	)

	clients_sorted_list, se_sorted_list = get_clients_and_ses_minimal_list(distance_lists_list,demotion_lists_list)

	pickle.dump(
		(
			get_complete_distance_matrix_1(distance_lists_list[0], demotion_lists_list[0], clients_sorted_list, se_sorted_list),
			get_complete_distance_matrix_1(distance_lists_list[1], demotion_lists_list[1], clients_sorted_list, se_sorted_list),
			get_complete_distance_matrix_1(distance_lists_list[2], demotion_lists_list[2], clients_sorted_list, se_sorted_list),
		),
		open(
			'whole_matrices.p',
			'wb'
		)
	)

def get_pca_of_the_matrices():
	'''
	Runs on the output of "dump_whole_matrices".
	Produces the principal components of the matrices and dumps only the top 80%.
	'''
	week_0_list, week_1_list, week_2_list = pickle.load(open('whole_matrices.p','rb'))

	if False:
		c = list(map(lambda e: e[1], week_0_list))\
				+ list(map(lambda e: e[1], week_1_list))\
				+ list(map(lambda e: e[1], week_2_list))

		l_0 = len(c[0])

		i = 0
		for plm in c:
			if len(plm) != l_0:
				print(
					str(i) + ' '\
					+ str(l_0) + ' '\
					+ str(plm))
				exit(0)
			i+=1

	if True:
		output_file = 'time_tags_and_top_80_pca_components.p'

	pca_train_array = np.array(
		list(map(lambda e: e[1], week_0_list))\
		+ list(map(lambda e: e[1], week_1_list))\
		+ list(map(lambda e: e[1], week_2_list))\

	)

	pca_engine = PCA()

	pca_engine.fit(pca_train_array)

	good_components_indexes_list = list()

	ratio_sum = 0
	for ind, r in sorted(enumerate(pca_engine.explained_variance_ratio_),key=lambda p: p[1],reverse=True):
		if ratio_sum >= 0.8:
			break
		ratio_sum += r
		good_components_indexes_list.append( ind )

	pca_array = pca_engine.transform( pca_train_array )

	# print('ratios list: ' + str(pca_engine.explained_variance_ratio_))

	print('numer of components: ' + str(good_components_indexes_list))

	pickle.dump(
		(
			tuple(
				map(
					lambda ind:\
						(
							week_0_list[ind][0],
							list(
								map(
									lambda pca_ind: pca_array[ind][pca_ind],
									good_components_indexes_list
								)
							),
						),
					range(len(week_0_list))
				)
			),
			tuple(
				map(
					lambda ind:\
						(
							week_1_list[ind][0],
							list(
								map(
									lambda pca_ind: pca_array[len(week_0_list) + ind][pca_ind],
									good_components_indexes_list
								)
							)
						),
					range(len(week_1_list))
				)
			),
			tuple(
				map(
					lambda ind:\
						(
							week_2_list[ind][0],
							list(
								map(
									lambda pca_ind: pca_array[len(week_0_list) + len(week_1_list) + ind][pca_ind],
									good_components_indexes_list
								)
							),
						),
					range(len(week_2_list))
				)
			),
		),
		open(
			PCA_DUMP_FOLDER + output_file,
			'wb'
		)
	)

def get_read_size_per_proc(i):
	d = dict()
	for tm , _ , _ , rs in json.load(open(filename_list[i][1],'rt')):
		if tm in d:
			d[tm] += rs
		else:
			d[tm] = rs
	return (filename_list[i][0],d,)

def get_read_size_per_time_moment_for_cern():
	global filename_list
	filename_list = list(
		itertools.chain.from_iterable(
			map(
				lambda p:\
					map(\
						lambda fn: (p[0],p[1] + fn,),
						os.listdir( p[1] )
					),
				enumerate(UNANSWERED_PATH_TUPLE)
			)
		)
	)

	week_dict_tuple = (dict(),dict(),dict(),)

	for ind,d in Pool(n_proc).map(get_read_size_per_proc, range(len(filename_list))):
		for tm in d.keys():
			if tm in week_dict_tuple[ind]:
				week_dict_tuple[ind][tm] += d[tm]
			else:
				week_dict_tuple[ind][tm] = d[tm]

	pickle.dump(
		(
			sorted(week_dict_tuple[0].items()),
			sorted(week_dict_tuple[1].items()),
			sorted(week_dict_tuple[2].items()),
		),
		open(
			PCA_DUMP_FOLDER + 'cern_viable_option_read_size.p',
			'wb'
		)
	)

def get_read_size_per_proc_1(i):
	'''
	Works on a portion of the unanswered queries.
	Returns a dictionary object of time_tag -> total_read_size.
	'''
	d = dict()
	for tm , rs in\
		map(
			lambda line: (int(line[0]),int(line[2]),),
			map(
				lambda line: line.split(','),
				file_variable[\
					g_idexes_pairs_tuple[i][0] : g_idexes_pairs_tuple[i][1]\
				]
			)
		):
		if tm in d:
			d[tm] += rs
		else:
			d[tm] = rs
	return d

def get_read_size_from_all_queries():
	'''
	Sets up data for transfer getting the dictionary of time_tag -> read_size for
	each of the three weeks.
	'''
	def get_rs_per_week(input_file_name):
		global file_variable, g_idexes_pairs_tuple

		file_variable = open(input_file_name,'rt').read()
		# len(file_variable) = 318007736

		file_variable = file_variable.split('\n')[1:]
		if len(file_variable[-1]) < 5: file_variable = file_variable[:-1]

		q_num = len(file_variable)

		g_idexes_pairs_tuple = []
		i = 0
		a = q_num // 200
		while i < q_num:
			g_idexes_pairs_tuple.append((
				i,
				i + a,
			))
			i+=a
		if g_idexes_pairs_tuple[-1][1] != q_num:
			g_idexes_pairs_tuple[-1] = (
				g_idexes_pairs_tuple[-1][0],
				q_num
			)

		print( 'Will start pool !' )

		p = Pool(n_proc)

		d_list = p.map(get_read_size_per_proc_1,range(len(g_idexes_pairs_tuple)))

		p.close()

		week_dict = dict()

		for d in d_list:
			for tm in d.keys():
				if tm in week_dict:
					week_dict[tm] += d[tm]
				else:
					week_dict[tm] = d[tm]

		return sorted( week_dict.items() )

	pickle.dump(
		(
			get_rs_per_week(RAW_QUERY_FILES[0]),
			get_rs_per_week(RAW_QUERY_FILES[1]),
			get_rs_per_week(RAW_QUERY_FILES[2]),
		),
		open(
			PCA_DUMP_FOLDER + 'all_queries_read_size.p',
			'wb'
		)
	)

def generate_throughput_trend_train():
	'''
	Puts whole matrices, average read size and throughput together based on time tags.
	'''
	thp_iterable_tuple = (
		get_thp('january_month_throughput.csv',WEEK_TIME_MOMENTS[0]),
		get_thp('february_month.csv',WEEK_TIME_MOMENTS[1]),
		get_thp('february_month.csv',WEEK_TIME_MOMENTS[2])
	)

	if False:
		rs_file = 'all_queries_read_size.p'
		data_set_file = 'all_q_rs_complete_data_set.p'

	if False:
		rs_file = 'cern_viable_option_read_size.p'
		data_set_file = 'complete_data_set.p'

	if True:
		rs_file = 'all_queries_read_size.p'
		matrices_file = 'time_tags_and_top_80_pca_components.p'
		data_set_file = 'complete_data_set_top_80.p'

	read_size_iterable_tuple = pickle.load(
		open(
			PCA_DUMP_FOLDER + rs_file,
			'rb'
		)
	)

	matrices_iterable = pickle.load(
		open(
			PCA_DUMP_FOLDER + matrices_file,
			'rb'
		)
	)

	pickle.dump(
		(
			get_training_set( thp_iterable_tuple[0] , read_size_iterable_tuple[0] , matrices_iterable[0] ),
			get_training_set( thp_iterable_tuple[1] , read_size_iterable_tuple[1] , matrices_iterable[1] ),
			get_training_set( thp_iterable_tuple[2] , read_size_iterable_tuple[2] , matrices_iterable[2] ),
		),
		open(
			PCA_DUMP_FOLDER + data_set_file,
			'wb'
		)
	)

def normalize_and_split_data_set(data_set_file,time_window):

	bins_list = get_normalized_values_per_week(data_set_file)

	indexes_dict = {
		( 0 , 0.05) : list(),
		( 0.05 , 0.1 ) : list(),
		( 0.1 , 0.15 ) : list(),
		( 0.15 , 0.2 ) : list(),
		( 0.2 , 0.25 ) : list(),
		( 0.25 , 0.3 ) : list(),
		( 0.3 , 0.35 ) : list(),
		( 0.35 , 0.4 ) : list(),
		( 0.4 , 0.45 ) : list(),
		( 0.45 , 0.5 ) : list(),
		( 0.5 , 0.55 ) : list(),
		( 0.55 , 0.6 ) : list(),
		( 0.6 , 0.65 ) : list(),
		( 0.65 , 0.7 ) : list(),
		( 0.7 , 0.75 ) : list(),
		( 0.75 , 0.8 ) : list(),
		( 0.8 , 0.85 ) : list(),
		( 0.85 , 0.9 ) : list(),
		( 0.9 , 0.95 ) : list(),
		( 0.95 , 1.05 ) : list(),
	}

	new_bin_list = list()

	i = 0
	for b_list in bins_list:
		for j in range(len(b_list)-time_window):

			new_bin_list.append(list())

			for l in range( j , j + time_window , 1 ):

				new_bin_list[-1].append(list())

				for e in b_list[ l ]:
					new_bin_list[-1][-1].append(e)

			for k in indexes_dict.keys():

				if k[0] <= b_list[ j + time_window - 1 ][-1] < k[1]:

					indexes_dict[k].append(i)

			i+=1

	bins_list = new_bin_list

	for k,v in indexes_dict.items():
		print(str(k) + ' ' + str(len(v)))

	valid_indexes_list = []
	for indexes_list in indexes_dict.values():
		if len(indexes_list) > 1:
			if len(indexes_list) == 2:
				valid_indexes_list.append(
					random.choice(indexes_list)
				)
			else:
				a = 0.2 * len(indexes_list)
				if a < 1:
					valid_indexes_list.append(
						random.choice(indexes_list)
					)
				else:
					valid_indexes_list += random.sample(
						indexes_list, round(a)
					)

	train_indexes_list = list(
		filter(lambda e: e not in valid_indexes_list, range(len(bins_list)))
	)

	print(len(train_indexes_list))

	print(len(valid_indexes_list))

	print(str(len(bins_list)) + ' ' + str(len(bins_list[0])) + ' ' + str(len(bins_list[0][0])))

	get_x_f = lambda indexes_l:\
	list(
		map(
			lambda p:\
				list(
					map(
						lambda e: e[:-1],
						p[1]
					)
				),
			filter(
				lambda p: p[0] in indexes_l,
				enumerate(bins_list)
			)
		)
	)
	get_y_f = lambda indexes_l:\
	list(
		map(
			lambda p:\
				list(
					map(
						lambda e: [e[-1],],
						p[1]
					)
				),
			filter(
				lambda p: p[0] in indexes_l,
				enumerate(bins_list)
			)
		)
	)

	train_x_list, train_y_list, valid_x_list, valid_y_list =\
		get_x_f(train_indexes_list),\
		get_y_f(train_indexes_list),\
		get_x_f(valid_indexes_list),\
		get_y_f(valid_indexes_list)

	get_lens_f=lambda l: str(len(l)) + ' ' + str(len(l[0])) + ' ' + str(len(l[0][0]))

	print(get_lens_f(train_x_list))
	print(get_lens_f(train_y_list))
	print(get_lens_f(valid_x_list))
	print(get_lens_f(valid_y_list))

	if False:
		exit(0)
		print('Resched end !')
		json.dump(
			(
				train_x_list,
				train_y_list,
				valid_x_list,
				valid_y_list,
			),
			open(
				PCA_DUMP_FOLDER + 'ready_to_train_data_set.json',
				'wt'
			)
		)

	if True:
		return np.array(train_x_list),\
			np.array(train_y_list),\
			np.array(valid_x_list),\
			np.array(valid_y_list)

def get_model_1(ws, bins_no, small_net_flag, lstm_at_end):
	inp_layer = keras.layers.Input(shape=(ws, bins_no,))

	if small_net_flag:
		x = inp_layer
	else:
		x = keras.layers.TimeDistributed(
			keras.layers.Dense(
				units=bins_no,
				activation='relu'
			)
		)(inp_layer)
		# x = keras.layers.LeakyReLU(alpha=0.3)(x)
		x = keras.layers.BatchNormalization()(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=bins_no,
			return_sequences=True,
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=10,
				activation='relu')
	)(x)
	# x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.BatchNormalization()(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=5,
				activation='relu')
	)(x)
	# x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.BatchNormalization()(x)

	if lstm_at_end:
		x = keras.layers.Bidirectional(
			keras.layers.LSTM(
				units=1,
				return_sequences=True,
			)
		)(x)
		x = keras.layers.BatchNormalization()(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=1, activation='sigmoid')
	)(x)

	# x = keras.layers.Dense(units=1, activation='relu')(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def train_main():
	def get_model_2(ws, bins_no):
		inp_layer = keras.layers.Input(shape=(ws, bins_no,))

		def get_residual_unit(input_layer, bins_no):

			x = keras.layers.Bidirectional(
				keras.layers.LSTM(
					units=bins_no,
					return_sequences=True,
				)
			)(input_layer)
			x = keras.layers.BatchNormalization()(x)

			x = keras.layers.TimeDistributed(
				keras.layers.Dense(units=bins_no,)
			)(x)
			x = keras.layers.LeakyReLU(alpha=0.3)(x)
			x = keras.layers.BatchNormalization()(x)

			return keras.layers.Add()([input_layer, x])

		x = get_residual_unit(inp_layer, bins_no)
		x = get_residual_unit(x, bins_no)
		x = get_residual_unit(x, bins_no)
		x = get_residual_unit(x, bins_no)

		x = keras.layers.TimeDistributed(
			keras.layers.Dense(units=10,)
		)(x)
		x = keras.layers.LeakyReLU(alpha=0.3)(x)
		x = keras.layers.BatchNormalization()(x)

		x = keras.layers.TimeDistributed(
			keras.layers.Dense(units=5,)
		)(x)
		x = keras.layers.LeakyReLU(alpha=0.3)(x)
		x = keras.layers.BatchNormalization()(x)

		x = keras.layers.TimeDistributed(
			keras.layers.Dense(units=1, activation='relu')
		)(x)

		return keras.models.Model(inputs=inp_layer, outputs=x)

	if False:
		X_train,\
		y_train,\
		X_valid,\
		y_valid = normalize_and_split_data_set('complete_data_set_top_80.p',30)
		out_folder = 'pca_models/'

	if False:
		X_train, y_train, X_valid, y_valid =\
		tuple(
			map(
				lambda e: np.array(e),
				json.load(
					open(
						PCA_DUMP_FOLDER+'ready_to_train_data_set.json',
						'rt'
					)
				)
			)
		)
		out_folder = 'pca_models/'

	if True:
		X_train,\
		y_train,\
		X_valid,\
		y_valid = normalize_and_split_data_set_1(
			'complete_data_set_top_80.p',
			40,
			39
		)

	# setting up training parameters
	small_net_flag = False
	optimizer_type = 0
		# 0 - default adam
		# 1 - adam b1=0.1 ; b2=0.1
		# 2 - adam b1=0.7 ; b2= 0.7
		# 3 - rmsprop 0.7
	run_with_lr_scheduler_flag = False
	random_train_validation_split_flag = False
	load_pretrained_flag = False
	number_of_epochs = 200
	batch_s = 64
	model_type = 1
	continuous_generation_flag = False
	models_dump_path = './pca_multiple_model_folders/models_12/'
	csv_log_path = './pca_csv_folder/losses_12.csv'

	def print_training_version():
		print('small_net_flag: ' + str(small_net_flag))
		print('optimizer_type: ' + str(optimizer_type))
		print('run_with_lr_scheduler_flag: ' + str(run_with_lr_scheduler_flag))
		print('random_train_validation_split_flag: ' + str(random_train_validation_split_flag))
		print('load_pretrained_flag: ' + str(load_pretrained_flag))
		print('number_of_epochs: ' + str(number_of_epochs))
		print('batch_size: ' + str(batch_s))
		print('model_type: ' + str(model_type))
		print('continuous_generation_flag: ' + str(continuous_generation_flag))

		print('models_dump_path: ' + str(models_dump_path))
		print('csv_log_path: ' + str(csv_log_path))

	def step_decay(epoch):
		initial_rate = 0.001
		new_rate = initial_rate
		for _ in range(epoch // 30):
			new_rate = new_rate * 0.9
		return new_rate
	lrate = keras.callbacks.LearningRateScheduler(step_decay)

	if True:
		tf.device('/gpu:0')

	if model_type == 1:
		model = get_model_1(X_train.shape[1], X_train.shape[2], small_net_flag)
	else:
		model = get_model_2(X_train.shape[1], X_train.shape[2])

	if optimizer_type == 0:
		model.compile(
			optimizer=keras.optimizers.Adam(),
			loss='mean_absolute_percentage_error',
			metrics=['mae',]
		)
	elif optimizer_type == 1:
		model.compile(
			optimizer=keras.optimizers.Adam(beta_1=0.1, beta_2=0.1),
			loss='mean_absolute_percentage_error',
			metrics=['mae',]
		)
	elif optimizer_type == 2:
		model.compile(
			optimizer=keras.optimizers.Adam(beta_1=0.7, beta_2=0.7),
			loss='mean_absolute_percentage_error',
			metrics=['mae',]
		)
	else:
		model.compile(
			optimizer=keras.optimizers.RMSprop(rho=0.7),
			loss='mean_absolute_percentage_error',
			metrics=['mae',]
		)

	if load_pretrained_flag:
		old_model = keras.models.load_model(
			'model_0248.hdf5')
		i = -1
		while i >= 0:
			if old_model.layers[i].name.startswith('bidirectional'):
				break
			if old_model.layers[i].name.startswith('time_distributed'):
				model.layers[i].set_weights(old_model.layers[i].get_weights())
			i-=1

	model.summary()

	if random_train_validation_split_flag:
		#shuffle train, valid
		a_list = list(range(X_train.shape[0]))
		to_shuffle_indexes_list = []
		while len(to_shuffle_indexes_list) < X_valid.shape[0]:
			to_shuffle_indexes_list.append(
				random.choice( a_list )
			)
			a_list.remove(to_shuffle_indexes_list[-1])
		x_array = np.empty( X_valid.shape[1:] )
		y_array = np.empty( y_valid.shape[1:] )

		for ii,ind in enumerate(to_shuffle_indexes_list):
			x_array = X_train[ind]
			X_train[ind] = X_valid[ii]
			X_valid[ii] = x_array

			y_array = y_train[ind]
			y_train[ind] = y_valid[ii]
			y_valid[ii] = y_array

	callbacks_list = [
		keras.callbacks.ModelCheckpoint(models_dump_path + "model_{epoch:04d}.hdf5", monitor='loss', save_best_only=True),
		keras.callbacks.CSVLogger(csv_log_path),
	]
	if run_with_lr_scheduler_flag:
		callbacks_list.append(lrate)

	print()
	print_training_version()
	# print(X_train.shape)
	# print(X_valid.shape)
	print()

	if not continuous_generation_flag:
		model.fit(
			x=X_train,
			y=y_train,
			batch_size=batch_s,
			epochs=number_of_epochs,
			validation_data=(\
				X_valid,
				y_valid,
			),
			# callbacks=callbacks_list
		)
	else:
		def generate_data_set(x_array, y_array, batch_size):
			current_index = 0
			batch_x_array = np.empty((
				batch_size,
				x_array.shape[1],
				x_array.shape[2]
			))
			batch_y_array = np.empty((
				batch_size,
				y_array.shape[1],
				y_array.shape[2]
			))
			while True:
				for i in range(batch_size):
					batch_x_array[i] = x_array[current_index]
					batch_y_array[i] = y_array[current_index]
					current_index = ( current_index + 1 ) % x_array.shape[0]

				yield batch_x_array,batch_y_array

		model.fit_generator(
			generate_data_set(X_train, y_train, batch_s),
			steps_per_epoch=424,
			epochs=200,
			validation_data=(\
				X_valid,
				y_valid,
			),
			callbacks=callbacks_list
		)

	print_training_version()

	# terminal 1: bigger network + adam + models_0 + losses_0.csv
	# terminal 2: bigger network + adam + random train/validation split + models_1 + losses_1.csv
	# terminal 3: bigger network + adam + learning rate scheduler (once 30 epochs) + models_2 + losses_2.csv
	# terminal 4: bigger network + adam + 10 time seq length + models_3 + losses_3.csv
	# terminal 5: bigger network + adam + 40 time seq length + models_4 + losses_4.csv
	# terminal 1: bigger network + adam + 20 time seq length + models_5 + losses_5.csv
	# terminal 1: bigger network + adam(0.7) + 30 time seq length + models_6 + losses_6.csv
	# terminal 2: bigger network + adam + 30 time seq length + 16 bs + models_7 + losses_7.csv
	# terminal 3: bigger network + adam + 30 time seq length + 64 bs + models_8 + losses_8.csv
	# terminal 4: bigger network + rmsprop(0.7) + 30 time seq length + 32 bs + models_9 + losses_9.csv
	# termnall 1: resid network + adam + models_10 + losses_10.csv
	# termnall 1: bigger network + adam + sequential batches + models_11 + losses_11.csv
	# termnall 1: bigger network + adam + max overlap 2 + models_12 + losses_12.csv
	# termnall 1: dense network + adam + models_13 + losses_13.csv

	# skylake cpu bigger network + adam
	# minsky3 gpu0 bigger network + adam(0,1 ; 0,1) ; nu mai GT vs thp
	# minsky3 gpu1 smaller network + adam + no BN after input ; nu mai fac GT vs thp
	# minsky tab2 gpu2 bigger network + adam + learning rate scheduler
	# skylake cpu bigger network + adam + random train/validation split

	# result_8.p: [9.937310579768065, 0.033096183]
	# result_10.p: [11.295326864338614, 0.03690919]
	# result_0.p: [11.541043326995052, 0.040025834]
	# result_4.p: [11.590152716674732, 0.04221123]
	# result_9.p: [12.168184057193939, 0.04608944]
	# result_7.p: [15.357107542605203, 0.059241258]
	# result_1.p: [13.025947358494646, 0.049118992]
	# result_6.p: [12.661994606120995, 0.044442177]
	# result_2.p: [12.320725220705405, 0.045876436]
	# result_5.p: [14.93303690304028, 0.057482604]
	# result_3.p: [13.21034426138444, 0.04825518]

def get_dataset_for_dense_net(data_set_file):
	examples_list = tuple(

		itertools.chain.from_iterable(

			get_normalized_values_per_week(data_set_file)

		)
	)

	valid_indexes_list = random.sample(
		range( len( examples_list ) ),
		round( 0.2 * len( examples_list ) ),
	)

	X_train_iterable, y_train_iterable, X_valid_iterable, y_valid_iterable =\
		list(),list(),list(),list()

	for ind, example in enumerate(examples_list):
		if ind in valid_indexes_list:
			X_train_iterable.append( example[:-1] )
			y_train_iterable.append( [ example[-1] , ] )
		else:
			X_valid_iterable.append( example[:-1] )
			y_valid_iterable.append( [ example[-1] , ] )

	del examples_list

	return\
		np.array(X_train_iterable),\
		np.array(y_train_iterable),\
		np.array(X_valid_iterable),\
		np.array(y_valid_iterable)

def train_main_0(data_set_file):
	X_train_iterable, y_train_iterable, X_valid_iterable, y_valid_iterable\
		=get_dataset_for_dense_net(data_set_file)

	inp_layer = keras.layers.Input(shape=(X_train_iterable.shape[-1],))
	x = keras.layers.Dense(units=12)(inp_layer)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dense(units=6)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dense(units=3)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dense(units=1, activation='relu')(x)
	model = keras.models.Model(inputs=inp_layer, outputs=x)

	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	model.fit(
		x=X_train_iterable,
		y=y_train_iterable,
		batch_size=64,
		epochs=100000,
		validation_data=(\
			X_valid_iterable,
			y_valid_iterable,
		),
		callbacks=[
			keras.callbacks.ModelCheckpoint('./pca_multiple_model_folders/models_13/' + "model_{epoch:04d}.hdf5", monitor='loss', save_best_only=True),
			keras.callbacks.CSVLogger('./pca_csv_folder/losses_13.csv'),
		]
	)

def get_results_0(data_set_file,best_model_path):

	model = keras.models.load_model( best_model_path )

	predict_list = list()

	all_x_list = list()
	all_y_list = list()

	for week_list in get_normalized_values_per_week(data_set_file):

		all_x_list +=\
		list(
			map(
				lambda e: e[:-1],
				week_list
			)
		)

		all_y_list +=\
		list(
			map(
				lambda e: [ e[-1] , ],
				week_list
			)
		)

		predict_list.append((
			list(
				map(
					lambda e: e[-1],
					week_list
				)
			),
			list(
				map(
					lambda e: e[0],
					model.predict(
						np.array(
							list(
								map(
									lambda e: e[:-1],
									week_list
								)
							)
						)
					)
				)
			)
		))

	pickle.dump(
		(
			(
				'performance',
				model.evaluate(
					np.array(all_x_list),
					np.array(all_y_list),
				)
			),
			(
				'predictions',
				predict_list
			),
		),
		open(
			'./pca_results/result_13.p',
			'wb'
		)
	)

def get_model_for_grid_search_0(model_creation_dict):

	model = get_model_1(
		model_creation_dict['sequence_length'],
		12,
		model_creation_dict['small_flag'],
		model_creation_dict['lstm_flag']
	)

	model.compile(
		optimizer=model_creation_dict['optimizer'],
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	return model

def get_model_for_grid_search_1(model_creation_dict):
	inp_layer = keras.layers.Input(shape=(12,))
	x = keras.layers.Dense(units=12)(inp_layer)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dense(units=6)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dense(units=3)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dense(units=1, activation='relu')(x)
	model = keras.models.Model(inputs=inp_layer, outputs=x)

	model.compile(
		optimizer=model_creation_dict['optimizer'],
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	return model

def get_model_for_grid_search_2(model_creation_dict):

	model = get_model_1(
		model_creation_dict['sequence_length'],
		12,
		True
	)

	model.compile(
		optimizer=model_creation_dict['optimizer'],
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	return model

def fit_model_0(model, model_fit_dict):

	def step_decay(epoch):
		initial_rate = 0.001
		new_rate = initial_rate
		for _ in range(epoch // model_fit_dict['epoch_interval']):
			new_rate = new_rate * 0.9
		return new_rate

	model.fit(
		x=data_set_dict[ model_fit_dict[ 'data_set_key' ] ][0],
		y=data_set_dict[ model_fit_dict[ 'data_set_key' ] ][1],
		batch_size=model_fit_dict['batch_size'],
		epochs=model_fit_dict['epochs'],
		validation_data=(\
			data_set_dict[ model_fit_dict[ 'data_set_key' ] ][2],
			data_set_dict[ model_fit_dict[ 'data_set_key' ] ][3],
		),
		verbose=0,
		callbacks=[
			keras.callbacks.CSVLogger(
				model_fit_dict['csv_log_path']
			),
			keras.callbacks.ModelCheckpoint(
				model_fit_dict['models_dump_path'] + "model_{epoch:04d}.hdf5",
				monitor='loss',
				save_best_only=True
			),
			keras.callbacks.LearningRateScheduler(step_decay),
		]
	)

def fit_model_1(model, model_fit_dict):
	model.fit(
		x=data_set_dict[ model_fit_dict[ 'data_set_key' ] ][0],
		y=data_set_dict[ model_fit_dict[ 'data_set_key' ] ][1],
		batch_size=model_fit_dict['batch_size'],
		epochs=model_fit_dict['epochs'],
		validation_data=(\
			data_set_dict[ model_fit_dict[ 'data_set_key' ] ][2],
			data_set_dict[ model_fit_dict[ 'data_set_key' ] ][3],
		),
		verbose=0,
		callbacks=[
			keras.callbacks.CSVLogger(
				model_fit_dict['csv_log_path']
			),
			keras.callbacks.ModelCheckpoint(
				model_fit_dict['models_dump_path'] + "model_{epoch:04d}.hdf5",
				monitor='val_loss',
				save_best_only=True
			),
			keras.callbacks.EarlyStopping(
				monitor='val_loss',
				patience=model_fit_dict['patience'],
			)
		]
	)

def nn_train_per_proc(i):
	model_create_f, create_dict, model_fit_f, fit_dict = pool_arguments_list[i]

	print('Entered training !')

	gpu_string = proc_q.get()

	print('\n\n\n\n' + str(os.getpid()) + ': will start training on ' + gpu_string + '\n\n\n\n')

	with tf.device(gpu_string):

		# with tf.compat.v1.Session() as sess:
		with tf.Session() as sess:

			# tf.compat.v1.keras.backend.set_session(sess)
			K.set_session

			model = model_create_f(create_dict)

			model_fit_f(model,fit_dict)

	print('\nFinished: ' + str(i) + ' ' + fit_dict['csv_log_path'])

	proc_q.put(gpu_string)

def get_lr_decay_arguments_pool(new_index):
	epochs_speis = 10

	pool_arguments_list = list()

	while epochs_speis <= 50:

		with open( 'pca_index_meaning/' + str(new_index) + '.txt' , 'wt' ) as myfile:
			myfile.write('bla bla')

		models_folder_name = 'models_' + str(new_index)

		if models_folder_name not in os.listdir('./pca_multiple_model_folders'):
			os.mkdir(
				'./pca_multiple_model_folders/'\
				+ models_folder_name
			)

		pool_arguments_list.append(
			(
				get_model_0,
				{
					'optimizer' : keras.optimizers.RMSprop(),
					'sequence_length' : 30,
				},
				fit_model_0,
				{
					'epoch_interval' : epochs_speis,
					'csv_log_path' :\
						'./pca_csv_folder/losses_'\
						+ str( new_index )
						+'.csv',
					'models_dump_path' :\
						'pca_multiple_model_folders/'\
						+ models_folder_name + '/',
					'data_set_key':'time_window_30',
					'batch_size':64,
					'epochs':1
				},
			)
		)

		new_index += 1

		epochs_speis += 10

	return pool_arguments_list

def get_overlap_parameters(new_index):
	pool_arguments_list = list()

	for overlap in range(6):

		with open( 'pca_index_meaning/' + str(new_index) + '.txt' , 'wt' ) as myfile:
			myfile.write(str(new_index) + ': default rmsprop + overlap ' + str(overlap))

		models_folder_name = 'models_' + str(new_index)

		if models_folder_name not in os.listdir('./pca_multiple_model_folders'):
			os.mkdir(
				'./pca_multiple_model_folders/'\
				+ models_folder_name
			)

		pool_arguments_list.append(
			(
				get_model_0,
				{
					'optimizer' : keras.optimizers.Adam(),
					'sequence_length' : 40,
				},
				fit_model_1,
				{
					'csv_log_path' :\
						'./pca_csv_folder/losses_'\
						+ str( new_index )
						+'.csv',
					'models_dump_path' :\
						'pca_multiple_model_folders/'\
						+ models_folder_name + '/',
					'data_set_key':'overlap_' + str(overlap),
					'batch_size':64,
					'epochs':200
				},
			)
		)

		new_index += 1

	return pool_arguments_list

def get_lr_vs_bs_parameters(new_index, data_set_name='normal_set'):
	pool_arguments_list = list()

	for l_r in (0.004, 0.001, 0.0004 ):

		for b_s in (256, 128, 64,):

			with open( 'pca_index_meaning/' + str(new_index) + '.txt' , 'wt' ) as myfile:
				myfile.write(\
					str(new_index)\
					+': default adam '\
					+'+ only dense layers '\
					+'+ lr ' + str(l_r) + ' '\
					+'+ batch size ' + str(b_s)
					)

			models_folder_name = 'models_' + str(new_index)

			if models_folder_name not in os.listdir('./pca_multiple_model_folders'):
				os.mkdir(
					'./pca_multiple_model_folders/'\
					+ models_folder_name
				)
			else:
				for to_del_fn in os.listdir('./pca_multiple_model_folders/'+ models_folder_name):
					os.remove('./pca_multiple_model_folders/'+ models_folder_name+'/'+to_del_fn)

			pool_arguments_list.append(
				(
					get_model_for_grid_search_1,
					{
						'optimizer' : keras.optimizers.Adam(lr=l_r),
					},
					fit_model_1,
					{
						'csv_log_path' :\
							'./pca_csv_folder/losses_'\
							+ str( new_index )
							+'.csv',
						'models_dump_path' :\
							'pca_multiple_model_folders/'\
							+ models_folder_name + '/',
						'data_set_key':data_set_name,
						'batch_size':b_s,
						'epochs':7000,
						'patience':1750,
					},
				)
			)

			pickle.dump(
				(
					0,
					pool_arguments_list[-1][1],
					1,
					pool_arguments_list[-1][3]
				),
				open(
					'pca_train_parameters/'+str(new_index)+'.p',
					'wb'
				)
			)

			new_index += 1

	return pool_arguments_list, new_index

def get_lr_vs_bs_parameters_1(
	new_index,
	data_set_name,
	get_log_str_function,
	get_model_function=get_model_for_grid_search_0,
	small_flag=False,
	lstm_flag=False,
	learning_rate_list=[],
	batch_size_list=[],
	already_trained_list=[]):

	pool_arguments_list = list()

	for l_r in learning_rate_list:

		for b_s in batch_size_list:

			if new_index not in already_trained_list:

				with open( 'pca_index_meaning/' + str(new_index) + '.txt' , 'wt' ) as myfile:
					myfile.write(
						get_log_str_function(
							new_index,
							l_r,
							b_s,
							data_set_name,
							small_flag,
							lstm_flag,
						)
					)

				models_folder_name = 'models_' + str(new_index)

				if models_folder_name not in os.listdir('./pca_multiple_model_folders'):
					os.mkdir(
						'./pca_multiple_model_folders/'\
						+ models_folder_name
					)
				else:
					for to_del_fn in os.listdir('./pca_multiple_model_folders/'+ models_folder_name):
						os.remove('./pca_multiple_model_folders/'+ models_folder_name+'/'+to_del_fn)

				pool_arguments_list.append(
					(
						get_model_function,
						{
							'optimizer' : keras.optimizers.Adam(lr=l_r),
							'sequence_length' : 40,
							'small_flag' : small_flag,
							'lstm_flag' : lstm_flag
						},
						fit_model_1,
						{
							'csv_log_path' :\
								'./pca_csv_folder/losses_'\
								+ str( new_index )
								+'.csv',
							'models_dump_path' :\
								'pca_multiple_model_folders/'\
								+ models_folder_name + '/',
							'data_set_key':data_set_name,
							'batch_size':b_s,
							'epochs':500,
							'patience':125,
						},
					)
				)

				pickle.dump(
					(
						0,
						pool_arguments_list[-1][1],
						1,
						pool_arguments_list[-1][3]
					),
					open(
						'pca_train_parameters/'+str(new_index)+'.p',
						'wb'
					)
				)

			new_index += 1

	return pool_arguments_list, new_index

def transform_array_only_last(arr):
	new_arr = np.empty( ( arr.shape[0] , 1 ) )
	for i in range( arr.shape[0] ): new_arr[i,0] = arr[i,-1,0]
	return new_arr

def hyper_parameters_sweep_train():
	# range: 284 - 293
	# range: 293 - 302

	models_folders_list = list()

	if False:
		new_index = max(
			map(
				lambda fn: int( fn.split('_')[1] ),
				os.listdir('./pca_multiple_model_folders')
			)
		)

		new_index += 1

		print(new_index)

		exit(0)
	if True:
		new_index = 293


	if False:
		data_set_index = max(
			map(
				lambda fn: int( fn.split('.')[0] ),
				os.listdir('./pca_data_sets/')
			)
		) + 1
	if True:
		data_set_index = 4

	global data_set_dict, pool_arguments_list, proc_q

# start: Produce data set
	data_set_dict = dict()

	if False:
		data_set_dict['time_window_30'] =\
		tuple(
			map(
				lambda e: np.array(e),
				json.load(
					open(
						PCA_DUMP_FOLDER+'ready_to_train_data_set.json',
						'rt'
					)
				)
			)
		)

	if False:
		for overlap in range(6):
			data_set_dict['overlap_' + str(overlap)] = normalize_and_split_data_set_1(
				'complete_data_set_top_80.p',
				40,
				overlap,
			)

	if False:
		data_set_dict['time_length_40'] = normalize_and_split_data_set_1(
			'complete_data_set_top_80.p',
			40,
			20,
		)

	if False:
		data_set_dict['normal_set'] = get_dataset_for_dense_net('complete_data_set_top_80.p')
		data_set_dict['overlap_39'] = normalize_and_split_data_set_1('complete_data_set_top_80.p',40,39)
		dump_data_sets_flag = True

	if False:
		data_set_dict['overlap_39'] = pickle.load(open('./pca_data_sets/5.p','rb'))[2]
		dump_data_sets_flag = False

	if False:
		tr_x, tr_y, va_x, va_y = pickle.load(open('./pca_data_sets/5.p','rb'))[2]
		tr_y, va_y =\
			transform_array_only_last(tr_y),\
			transform_array_only_last(va_y)
		data_set_dict['overlap_39'] = ( tr_x, tr_y, va_x, va_y, )
		dump_data_sets_flag = False

	if False:
		data_set_dict['normal_set'] = pickle.load(open('./pca_data_sets/1.p','rb'))[2]
		data_set_dict['overlap_10'] = pickle.load(open('./pca_data_sets/2.p','rb'))[2]
		data_set_dict['overlap_20'] = pickle.load(open('./pca_data_sets/0.p','rb'))[2]
		dump_data_sets_flag = False

	if True:
		data_set_dict['normal_set'] = pickle.load(open('./pca_data_sets/4.p','rb'))[2]
		data_set_dict['normal_set'] =\
		(
			data_set_dict['normal_set'][ 2 ],
			data_set_dict['normal_set'][ 3 ],
			data_set_dict['normal_set'][ 0 ],
			data_set_dict['normal_set'][ 1 ],
		)
		dump_data_sets_flag = False

		print('\n\n\n\n Will train on: ' + str(data_set_dict['normal_set'][0].shape) + '\n\n\n\n')

# end: Produce data set

	index_copy = new_index

# start: Produce arguments
	pool_arguments_list = list()

	# a, new_index = get_lr_vs_bs_parameters(
	# 	new_index,
	# 	'normal_set'
	# )
	# pool_arguments_list += a

	# pool_arguments_list, new_index = get_lr_vs_bs_parameters_1(new_index, 'overlap_10')

	already_trained = (
		230 , 231 , 232 , 233, 235 , 237
	)

	log_func = lambda ind, lr, bs, data_set_name, small_flag, lstm_flag:\
		str(ind)\
		+': default adam '\
		+ ( '+ time distributed' if not small_flag else '+ lstm' ) + ' at begining '\
		+ ( '+ lstm' if lstm_flag else '+ time distributed' ) + ' at end '\
		+'+ data_set_name ' + str( data_set_name ) + ' only last cell '\
		+'+ lr ' + str(lr) + ' '\
		+'+ batch size ' + str(bs)

	if True:
		a, new_index = get_lr_vs_bs_parameters(new_index)
		pool_arguments_list += a

	if False:
		# start 26th april, start_index=284
		lr_list = ( 0.004 , 0.001, 0.0004 )
		bs_list = ( 256 , 128 , 64 )
		a, new_index = get_lr_vs_bs_parameters_1(
			new_index,
			'overlap_39',
			log_func,
			get_model_for_grid_search_0,
			False,
			True,
			lr_list,
			bs_list,
			already_trained
		)
		pool_arguments_list += a

	if False:
		# 25th april, first_index=250 , last_config_index=283
		a, new_index = get_lr_vs_bs_parameters_1(
			new_index,
			'overlap_39',
			log_func,
			get_model_for_grid_search_0,
			True,
			True,
			(0.001,),
			(64,),
			already_trained
		)
		pool_arguments_list += a

		lr_list = ( 0.004 , 0.0004 )
		bs_list = ( 256 , 128 , 64 )
		a, new_index = get_lr_vs_bs_parameters_1(
			new_index,
			'overlap_39',
			log_func,
			get_model_for_grid_search_0,
			True,
			False,
			lr_list,
			bs_list,
			already_trained
		)
		pool_arguments_list += a
		a, new_index = get_lr_vs_bs_parameters_1(
			new_index,
			'overlap_39',
			log_func,
			get_model_for_grid_search_0,
			False,
			False,
			lr_list,
			bs_list,
			already_trained
		)
		pool_arguments_list += a
		a, new_index = get_lr_vs_bs_parameters_1(
			new_index,
			'overlap_39',
			log_func,
			get_model_for_grid_search_0,
			False,
			True,
			lr_list,
			bs_list,
			already_trained
		)
		pool_arguments_list += a
		a, new_index = get_lr_vs_bs_parameters_1(
			new_index,
			'overlap_39',
			log_func,
			get_model_for_grid_search_0,
			True,
			True,
			lr_list,
			bs_list,
			already_trained
		)
		pool_arguments_list += a

		lr_list = ( 0.001, )
		bs_list = ( 256, )
		a, new_index = get_lr_vs_bs_parameters_1(
			new_index,
			'overlap_39',
			log_func,
			get_model_for_grid_search_0,
			True,
			False,
			lr_list,
			bs_list,
			already_trained
		)
		pool_arguments_list += a
		a, new_index = get_lr_vs_bs_parameters_1(
			new_index,
			'overlap_39',
			log_func,
			get_model_for_grid_search_0,
			False,
			False,
			lr_list,
			bs_list,
			already_trained
		)
		pool_arguments_list += a
		a, new_index = get_lr_vs_bs_parameters_1(
			new_index,
			'overlap_39',
			log_func,
			get_model_for_grid_search_0,
			False,
			True,
			lr_list,
			bs_list,
			already_trained
		)
		pool_arguments_list += a
		a, new_index = get_lr_vs_bs_parameters_1(
			new_index,
			'overlap_39',
			log_func,
			get_model_for_grid_search_0,
			True,
			True,
			lr_list,
			bs_list,
			already_trained
		)
		pool_arguments_list += a

	# finished for: 214 222 206 207 223 215 230 216 208 224

	print('\n\n\n\n\nlast index: ' + str(new_index) + '\n\n\n\n\n')

# end: Produce arguments

# start: Dump data_sets
	if dump_data_sets_flag:
		for data_set_label, ds in data_set_dict.items():

			pickle.dump(
				(
					data_set_label,
					list(
						map(
							lambda e: e[1],
							filter(
								lambda arg: arg[0][3]['data_set_key'] == data_set_label,
								zip(
									pool_arguments_list,
									range(index_copy, new_index)
								)
							)
						)
					),
					ds,
				),
				open(
					'./pca_data_sets/' + str(data_set_index) + '.p',
					'wb'
				)
			)

			data_set_index += 1

#end: Dump data_sets

	with open('pca_what_is_what.txt', 'wt') as myfile:
		for _, content in sorted( map(
			lambda fn: ( int(fn[:-4]) ,  open('pca_index_meaning/'+fn).read() ,),
			os.listdir( 'pca_index_meaning' ) ) ):
				myfile.write(content + '\n')

	proc_q = Queue()
	a = 0
	# for _ in range(3):
	for gpu_string in ('/gpu:0','/gpu:1','/gpu:2','/gpu:3'):
		proc_q.put( gpu_string )
		a+=1

	print('Will start pool !')

	Pool( a ).map(
		nn_train_per_proc,
		range( len(pool_arguments_list) )
	)

Cache_Element = namedtuple(
	'Cache_El',
	[
		'x_per_week',
		'y_per_week',
		'x_all_weeks',
		'y_all_weeks',
		'x_valid',
		'y_valid'
	]
)

def set_up_windowed_data_set(bins_list, time_window, max_overlap_length=39):
	x_iterables_list = list()
	y_iterables_list = list()
	for b_list in bins_list:

		x_iterable = list()
		y_iterable = list()

		last_appended_example_index = -2000

		for j in range(len(b_list)-time_window):

			if time_window - j + last_appended_example_index <= max_overlap_length:

				last_appended_example_index = j

				x_iterable.append(list())
				y_iterable.append(list())

				for l in range( j , j + time_window , 1 ):

					x_iterable[-1].append(list())
					y_iterable[-1].append([b_list[ l ][ - 1 ],])

					for e in b_list[ l ][:-1]:
						x_iterable[-1][-1].append(e)

		x_iterables_list.append( np.array( x_iterable ) )
		y_iterables_list.append( np.array( y_iterable ) )

	return x_iterables_list , y_iterables_list

def aggregate_arrays(x_iterables_list, y_iterables_list):
	x_array = np.empty( (
		sum(map( lambda arr: arr.shape[0] , x_iterables_list )),
		x_iterables_list[0].shape[1],
		x_iterables_list[0].shape[2],
	) )

	y_array = np.empty( (
		x_array.shape[0],
		x_array.shape[1],
		1,
	) )

	index_acc = 0

	for x_arr, y_arr in zip( x_iterables_list , y_iterables_list ):
		x_array[index_acc : index_acc + x_arr.shape[0]] = x_arr
		y_array[index_acc : index_acc + y_arr.shape[0]] = y_arr

		index_acc += x_arr.shape[0]

	return x_array, y_array

def get_best_valid_index_model(index):
	dumped_models_set = set(\
		map(
				lambda fn: int( fn[6:10] ),
				os.listdir(
					'./pca_multiple_model_folders/models_' + str(index)
				)
		)
	)

	if len(dumped_models_set) <= 3:
		return None

	g = csv.reader( open( './pca_csv_folder/' + 'losses_' + str( index ) +'.csv' , 'rt' ) )

	next(g)

	best_valid_model_and_loss = next(g)

	best_valid_model_and_loss = (\
		int(best_valid_model_and_loss[0]),
		float(best_valid_model_and_loss[-2]),
		best_valid_model_and_loss,
	)

	for line_list in g:
		a = int(line_list[0])

		if a in dumped_models_set:

			val_loss = float(line_list[-2])

			if val_loss < best_valid_model_and_loss[1]:
				best_valid_model_and_loss = (
					a,
					val_loss,
					line_list,
				)

	return best_valid_model_and_loss

def get_models_name_on_index(index):
	if index < 10:
		return 'model_000' + str(index) + '.hdf5'
	if index < 100:
		return 'model_00' + str(index) + '.hdf5'
	if index < 1000:
		return 'model_0' + str(index) + '.hdf5'
	return 'model_' + str(index) + '.hdf5'

def reorder_valid_in_temporal_order(all_x,all_y,v_x,v_y):
	indexes_list = list()

	for i in range( v_x.shape[0] ):

		for j in range( all_x.shape[0] ):

			if np.all( v_x[i] == all_x[j] ) and  np.all( v_y[i] == all_y[j] ):

				indexes_list.append( ( i , j ) )

				break

	indexes_list.sort(key=lambda k: k[1])

	new_x, new_y = np.empty( v_x.shape ), np.empty( v_y.shape )

	return_list = list()

	for ind, p in enumerate( indexes_list ):

		return_list.append( p[1] )

		new_x[ind] = v_x[ p[0] ]
		new_y[ind] = v_y[ p[0] ]

	return new_x, new_y, return_list

def aggregate_arrays_0(x_iterable, y_iterable):
	a_x = np.copy(
		x_iterable[0]
	)
	a_y = np.copy(
		y_iterable[0]
	)
	for i in range(1, len( x_iterable )):
		a_x = np.append(
			a_x,
			x_iterable[i],
			axis=0
		)
		a_y = np.append(
			a_y,
			y_iterable[i],
			axis=0
		)
	return a_x, a_y

def dump_nn_results(aaaaa):
	data_set_file, gpu_string, limits = aaaaa

	dir_list = os.listdir('pca_multiple_model_folders/')

	bins_list = get_normalized_values_per_week(data_set_file)

	time_window_cache_dict = dict()

	Cache_Element = namedtuple(
		'Cache_El',
		[
			'x_per_week',
			'y_per_week',
			'x_all_weeks',
			'y_all_weeks',
		]
	)

	evaluate_dict = dict()

	if False:
		for fn in os.listdir( './pca_data_sets/' ):
			_, indexes_list, ds = pickle.load( open( './pca_data_sets/' + fn , 'rb' ) )
			for ind in indexes_list:
				evaluate_dict[ind] = ( ds[2] , ds[3] , )

	if False:
		lines_in_what_is_what_list = list(
			filter(
				lambda line: len(line) > 2 and limits[0] <= int(line.split(':')[0]) < limits[1],
				open('./pca_what_is_what.txt','rt').read().split('\n')
			)
		)

		for fn in os.listdir( './pca_data_sets/' ):
			data_set_name, indexes_list, ds = pickle.load( open( './pca_data_sets/' + fn , 'rb' ) )

			if data_set_name == 'normal_set':
				for line in lines_in_what_is_what_list:
					if 'only dense layers' in line:
						evaluate_dict[ int(line.split(':')[0]) ] = ( ds[2] , ds[3] , )
			else:
				for line in lines_in_what_is_what_list:
					if data_set_name in line:
						evaluate_dict[ int(line.split(':')[0]) ] = ( ds[2] , ds[3] , )

	if True:
		normal_set = pickle.load(open( './pca_data_sets/4.p' , 'rb' ))[2][2:]
		overlap_39 = pickle.load(open( './pca_data_sets/5.p' , 'rb' ))[2][2:]

		for ind in range(206, 230):
			evaluate_dict[ind] = normal_set
		for ind in range(206, 260):
			evaluate_dict[ind] = overlap_39



	dir_list = list(filter( lambda dn: limits[0] <= int( dn.split('_')[1] ) < limits[1], os.listdir('pca_multiple_model_folders') ))

	with tf.device(gpu_string):
		with tf.compat.v1.Session() as sess:
			# with tf.Session() as sess:

			# tf.compat.v1.keras.backend.set_session(sess)
			K.set_session(sess)

			# for dir_name in dir_list:
			# for dir_name in ['models_12',]:
			for inddd,dir_name in enumerate(dir_list):

				print('is at: ' + str(inddd+1) + '/' + str(len(dir_list)))

				models_list = sorted(os.listdir('pca_multiple_model_folders/' + dir_name))

				if len(models_list) > 0:

					print( dir_name + ' ' + models_list[-1] )

					# model = keras.models.load_model(
					# 	'pca_multiple_model_folders/'\
					# 	+ dir_name + '/' + models_list[-1]
					# )

					model = keras.models.load_model(
						'pca_multiple_model_folders/'\
						+ dir_name + '/'\
						+ get_models_name_on_index( get_best_valid_index_model( int(dir_name.split('_')[1]) ) )
					)

					input_shape = model.layers[0].output_shape[0]

					print(input_shape)

					if len(input_shape) == 3:
						if input_shape[1] in time_window_cache_dict:
							cache_el = time_window_cache_dict[input_shape[1]]
						else:
							a, b = set_up_windowed_data_set(input_shape[1])

							c, d = aggregate_arrays(a,b)

							cache_el = Cache_Element(a,b,c,d)

							time_window_cache_dict[input_shape[1]] = cache_el
					else:
						if 1 in time_window_cache_dict:
							cache_el = time_window_cache_dict[1]
						else:
							cache_el = Cache_Element(
								list(
									map(
										lambda b_list:
											np.array(
												list(
													map(
														lambda ex: ex[:-1],
														b_list
													)
												)
											),
										bins_list
									)
								),
								list(
									map(
										lambda b_list:
											np.array(
												list(
													map(
														lambda ex: [ ex[-1] , ],
														b_list
													)
												)
											),
										bins_list
									)
								),
								np.array(
									list(
										itertools.chain.from_iterable(
											map(
												lambda b_list:
													map(
														lambda ex: ex[:-1],
														b_list
													),
												bins_list
											)
										)
									)
								),
								np.array(
									list(
										itertools.chain.from_iterable(
											map(
												lambda b_list:
													map(
														lambda ex: [ ex[-1] , ],
														b_list
													),
												bins_list
											)
										)
									)
								)
							)
							time_window_cache_dict[1] = cache_el

					predict_list = list()

					for week_ind in range( len( cache_el.x_per_week ) ):

						pred_array = model.predict(
							cache_el.x_per_week[ week_ind ]
						)

						if len(input_shape) == 3:

							week_pred_list = list()
							week_gt_list = list()

							for i in range( cache_el.x_per_week[ week_ind ].shape[1] - 1 ):

								week_pred_list.append( pred_array[0][i][0] )

								week_gt_list.append(
									cache_el.y_per_week[ week_ind ][0][i][0]
								)

							for i in range( cache_el.x_per_week[ week_ind ].shape[0] ):

								week_pred_list.append( pred_array[i][-1][0] )

								week_gt_list.append(
									cache_el.y_per_week[ week_ind ][i][-1][0]
								)

						else:
							week_gt_list = list(
								map(
									lambda l: l[0],
									cache_el.y_per_week[ week_ind ]
								)
							)
							week_pred_list = list(
								map(
									lambda l: l[0],
									pred_array
								)
							)

						predict_list.append(
							(
								week_gt_list,
								week_pred_list,
							)
						)

					dir_name_index = int( dir_name.split('_')[1] )

					pickle.dump(
						(
							(
								'general_performance',
								model.evaluate(
									cache_el.x_all_weeks,
									cache_el.y_all_weeks,
								)
							),
							(
								'validation_performance',
								model.evaluate(
									evaluate_dict[dir_name_index][0],
									evaluate_dict[dir_name_index][1],
								)
							),
							(
								'predictions',
								predict_list
							),
						),
						open(
							'./pca_results/result_' + dir_name.split('_')[1] + '.p',
							'wb'
						)
					)

					print()

def dump_nn_results_0(gpu_string):
	# with tf.device(gpu_string):
	# 	# with tf.compat.v1.Session() as sess:
	# 	with tf.Session() as sess:

	# 		# tf.compat.v1.keras.backend.set_session(sess)
	# 		K.set_session(sess)

	# 		for ind in gpu_task_dict[gpu_string]:
	if True:
		if True:
			for ind in (304,):

				best_valid_tuple = get_best_valid_index_model( ind )

				# model = keras.models.load_model(
				# 	'pca_multiple_model_folders/'\
				# 	+ 'models_' + str(ind) + '/'\
				# 	+ get_models_name_on_index( best_valid_tuple[0] )
				# )

				model = keras.models.load_model(
					'pca_multiple_model_folders/'\
					+ 'models_304/model_0318.hdf5'
				)

				input_shape = model.layers[0].output_shape[0]
				output_shape = model.layers[-1].output_shape

				# print('\n\n\n\n\n\n\n\n')
				# print( input_shape )
				# print( output_shape )
				# print( model.layers[-1].name )
				# print('\n\n\n\n\n\n\n\n')
				# return

				predict_list = list()

				for week_ind in range( len( index_data_set_dict[ind].x_per_week ) ):

					pred_array = model.predict(
						index_data_set_dict[ind].x_per_week[ week_ind ]
					)

					if len(input_shape) == 3 and len(output_shape) == 3:

						week_pred_list = list()
						week_gt_list = list()

						for i in range( index_data_set_dict[ind].x_per_week[ week_ind ].shape[1] - 1 ):

							week_pred_list.append( pred_array[0][i][0] )

							week_gt_list.append(
								index_data_set_dict[ind].y_per_week[ week_ind ][0][i][0]
							)

						for i in range( index_data_set_dict[ind].x_per_week[ week_ind ].shape[0] ):

							week_pred_list.append( pred_array[i][-1][0] )

							week_gt_list.append(
								index_data_set_dict[ind].y_per_week[ week_ind ][i][-1][0]
							)

					else:

						week_gt_list = list(
							map(
								lambda l: l[0],
								index_data_set_dict[ind].y_per_week[ week_ind ]
							)
						)
						week_pred_list = list(
							map(
								lambda l: l[0],
								pred_array
							)
						)

					predict_list.append(
						(
							week_gt_list,
							week_pred_list,
						)
					)

				if len(input_shape) == 3 and len(output_shape) == 3:

					valid_gt_list = list(
							map(
								lambda e: e[-1][0],
								index_data_set_dict[ind].y_valid
							)
						)

					valid_pred_list = list(
							map(
								lambda e: e[-1][0],
								model.predict(
									index_data_set_dict[ind].x_valid,
								)
							)
						)

				else:

					print(index_data_set_dict[ind].x_valid.shape)
					print()

					valid_gt_list = list(
							map(
								lambda e: e[0],
								index_data_set_dict[ind].y_valid
							)
						)

					valid_pred_list = list(
							map(
								lambda e: e[0],
								model.predict(
									index_data_set_dict[ind].x_valid,
								)
							)
						)

				pickle.dump(
					{
						'whole_data_set_performance' :\
							model.evaluate(
								index_data_set_dict[ind].x_all_weeks,
								index_data_set_dict[ind].y_all_weeks,
							),
						# 'best_model_performance' :\
						# 	best_valid_tuple[2],
						'best_model_performance' :\
							'318,6.942901716419845,0.026840016,7.137319571097857,0.024455104',
						'per_week_predictions' :\
							predict_list,
						'validation_predictions' :\
							(
								valid_gt_list,
								valid_pred_list

							)
					}					,
					open(
						'./pca_results/result_' + str(ind) + '.p',
						'wb'
					)
				)

# dump_nn_results_0('/gpu:1')
# exit(0)

def dump_results_main():
# Start: Set Up Dictionaries
	index_to_valid_dict,index_to_x_y_per_week_dict,x_y_per_week_dict,valid_set_dict,\
		x_y_whole_dict = dict(), dict(), dict(), dict(), dict()

	if False:
		bins_list = get_normalized_values_per_week('complete_data_set_top_80.p')
	if True:
		bins_list = get_normalized_values_per_week('tm_tren_ars_pc_per_week_4_may.p')

	if False:
		# best dense: 206
		# best RNN with relu: 242
		# best RNN with sigmoid: 283

		# x_y_per_week_dict[ 'only_dense' ] =\
		# 	(
		# 		list(
		# 			map(
		# 				lambda b_list:
		# 					np.array(
		# 						list(
		# 							map(
		# 								lambda ex: ex[:-1],
		# 								b_list
		# 							)
		# 						)
		# 					),
		# 				bins_list
		# 			)
		# 		),
		# 		list(
		# 			map(
		# 				lambda b_list:
		# 					np.array(
		# 						list(
		# 							map(
		# 								lambda ex: [ ex[-1] , ],
		# 								b_list
		# 							)
		# 						)
		# 					),
		# 				bins_list
		# 			)
		# 		)
		# 	)
		x_y_per_week_dict[ 'overlap_39' ] =\
			set_up_windowed_data_set(
				bins_list,
				40
			)

		# x_y_whole_dict[ 'only_dense' ] =\
		# 	aggregate_arrays_0(
		# 		x_y_per_week_dict[ 'only_dense' ][0],
		# 		x_y_per_week_dict[ 'only_dense' ][1],
		# 	)
		x_y_whole_dict[ 'overlap_39' ] =\
			aggregate_arrays_0(
				x_y_per_week_dict[ 'overlap_39' ][0],
				x_y_per_week_dict[ 'overlap_39' ][1],
			)

		a, b = pickle.load(open( './pca_data_sets/4.p' , 'rb' ))[2][2:]
		valid_set_dict[ 'only_dense' ] =\
			reorder_valid_in_temporal_order(
				x_y_whole_dict[ 'only_dense' ][0],
				x_y_whole_dict[ 'only_dense' ][1],
				a,
				b,
			)[:2]

		a, b = pickle.load(open( './pca_data_sets/5.p' , 'rb' ))[2][2:]
		valid_set_dict[ 'overlap_39' ] =\
			reorder_valid_in_temporal_order(
				x_y_whole_dict[ 'overlap_39' ][0],
				x_y_whole_dict[ 'overlap_39' ][1],
				a,
				b,
			)[:2]

		index_to_valid_dict[ 206 ] = 'only_dense'
		index_to_valid_dict[ 242 ] = 'overlap_39'
		# index_to_valid_dict[ 283 ] = 'overlap_39'

		index_to_x_y_per_week_dict.update( index_to_valid_dict )

	if True:
		x_y_per_week_dict[ 'overlap_39' ] =\
			set_up_windowed_data_set(
				bins_list,
				40
			)
		x_y_per_week_dict[ 'overlap_39' ] =\
		(
			x_y_per_week_dict[ 'overlap_39' ][ 0 ],
			tuple(
				map(
					lambda arr: transform_array_only_last(arr),
					x_y_per_week_dict[ 'overlap_39' ][ 1 ]
				)
			),
		)

		x_y_whole_dict[ 'overlap_39' ] =\
			aggregate_arrays_0(
				x_y_per_week_dict[ 'overlap_39' ][0],
				x_y_per_week_dict[ 'overlap_39' ][1],
			)

		a, b = pickle.load(open( './pca_data_sets/6.p' , 'rb' ))['train_valid_data_sets'][2:]
		valid_set_dict[ 'overlap_39' ] =\
			reorder_valid_in_temporal_order(
				x_y_whole_dict[ 'overlap_39' ][0],
				x_y_whole_dict[ 'overlap_39' ][1],
				a,
				transform_array_only_last(b),
			)[:2]

		index_to_valid_dict[ 304 ] = 'overlap_39'

		index_to_x_y_per_week_dict.update( index_to_valid_dict )

# End: Set Up Dictionaries

	global index_data_set_dict

	index_data_set_dict = dict()

	for index, valid_name in index_to_valid_dict.items():

		index_data_set_dict[index] =\
			Cache_Element(
				x_y_per_week_dict[ index_to_x_y_per_week_dict[ index ] ][0],
				x_y_per_week_dict[ index_to_x_y_per_week_dict[ index ] ][1],
				x_y_whole_dict[ index_to_x_y_per_week_dict[ index ] ][0],
				x_y_whole_dict[ index_to_x_y_per_week_dict[ index ] ][1],
				valid_set_dict[ valid_name ][0],
				valid_set_dict[ valid_name ][1],
			)

	print('\n\n\nFinished preparing data !\n\n\n')

# Start: Set up of available GPUs

	gpu_tuple = ('/gpu:2',)

# End: Set up of available GPUS
	global gpu_task_dict

	gpu_task_dict = dict()
	for g_s in gpu_tuple: gpu_task_dict[g_s] = list()

	gpu_i = 0

	for ind in index_data_set_dict.keys():

		gpu_task_dict[ gpu_tuple[gpu_i] ].append( ind )

		gpu_i = (gpu_i + 1) % len( gpu_tuple )

	Pool(len(gpu_tuple)).map(
		dump_nn_results_0,
		gpu_tuple
	)

def analyse_data_set():

	matrices_per_week_lists = pickle.load(open('./pca_dumps/whole_matrices.p','rb'))

	# print(matrices_per_week_lists[0][0])

	print(len(matrices_per_week_lists))

	print(\
		str(len(matrices_per_week_lists[0])) + ' '\
		+ str(len(matrices_per_week_lists[1])) + ' '\
		+ str(len(matrices_per_week_lists[2]))
	)

	print( len(matrices_per_week_lists[0][0]) )

	# reporting_time_diffs_per_week, cell_diffs_per_week =\
	# 	list(), list()

	# for ii,matrix_week in enumerate(matrices_per_week_lists):

	# 	time_diff_list = list()
	# 	cells_diff = list()

	# 	for i in range( 1 , len(matrix_week) ):

	# 		if WEEK_TIME_MOMENTS[ii][0] <= matrix_week[i][0] < WEEK_TIME_MOMENTS[ii][1]:

	# 			time_diff_list.append(
	# 				matrix_week[i][0] - matrix_week[i-1][0]
	# 			)

	# 			cells_diff.append(0)

	# 			for p in zip( matrix_week[i][1] , matrix_week[i-1][1] ):
	# 				if p[0] != p[1]:
	# 					cells_diff[-1] += 1

	# 	print('Finished week !')

	# 	reporting_time_diffs_per_week.append( time_diff_list )

	# 	cell_diffs_per_week.append( cells_diff )

	reporting_time_diffs_per_week, cell_diffs_per_week=\
	tuple(
		zip(\
			*map(
				lambda week_list:\
					zip(\
						*map(
							lambda p:\
								(
									p[1][0] - p[0][0],
									sum(
										map(
											lambda pp: pp[0] != pp[1],
											zip(
												p[0][1],
												p[1][1],
											)
										)
									)
								),
							zip(
								week_list[:-1],
								week_list[1:],
							)
						)
					)
				,
				matrices_per_week_lists
			)
		)
	)

	print( reporting_time_diffs_per_week[0][:10] )
	print( cell_diffs_per_week[0][:10] )

	pickle.dump(
		(
			reporting_time_diffs_per_week,
			cell_diffs_per_week
		),
		open(
			'pipe_1.p',
			'wb'
		)
	)

def test_generator():

	def generate_data_set(x_array, y_array, batch_size):
		current_index = 0
		batch_x_array = np.empty((
			batch_size,
			x_array.shape[1],
			x_array.shape[2]
		))
		batch_y_array = np.empty((
			batch_size,
			y_array.shape[1],
			y_array.shape[2]
		))
		while True:
			for i in range(batch_size):
				batch_x_array[i] = x_array[current_index]
				batch_y_array[i] = y_array[current_index]
				current_index = ( current_index + 1 ) % x_array.shape[0]

			yield batch_x_array,batch_y_array

	X_train, y_train, X_valid, y_valid =\
	tuple(
		map(
			lambda e: np.array(e),
			json.load(
				open(
					PCA_DUMP_FOLDER+'ready_to_train_data_set.json',
					'rt'
				)
			)
		)
	)

	g = generate_data_set( X_train , y_train , 2 )

	a, b = next( g )

	print( np.all(a == X_train[:2] ))
	print( np.all(b == y_train[:2] ))

	a, b = next( g )

	print( np.all(a == X_train[2:4] ))
	print( np.all(b == y_train[2:4] ))

def dump_pca_info_for_plot():
	week_0_list, week_1_list, week_2_list = pickle.load(open('./pca_dumps/whole_matrices.p','rb'))

	pca_train_array = np.array(
		list(map(lambda e: e[1], week_0_list))\
		+ list(map(lambda e: e[1], week_1_list))\
		+ list(map(lambda e: e[1], week_2_list))\

	)

	pca_engine = PCA()

	pca_engine.fit(pca_train_array)

	try:

		print(pca_engine.explained_variance_ratio_[:20])

		print(pca_engine.explained_variance_ratio_.cumsum())

	except:

		print('exception !')

	pickle.dump(
		pca_engine,
		open(
			'pipe.p',
			'wb'
		)
	)

	if False:

		# pickle.dump(
		# 	pca_engine.explained_variance_ratio_ ,
		# 	open(
		# 		'pipe.p',
		# 		'wb'
		# 	)
		# )

		pca_array = pca_engine.transform( pca_train_array )[:,:11]

		with open('components.csv','wt') as pc_f:

			for i in range(pca_array.shape[1]-1):
				pc_f.write( 'principal_component_' + str(i+1) + ',' )

			pc_f.write('principal_component_'+str(pca_array.shape[1])+'\n')

			for i in range(pca_array.shape[0]):
				for j in range(pca_array.shape[1]-1):
					pc_f.write( str(pca_array[i,j]) + ',' )
				pc_f.write( str(pca_array[i,j]) + '\n' )

def plot_components_from_csv(ind):
	a = csv.reader( open( 'components.csv' , 'rt' ) )
	next(a)
	a = list(a)
	if len( a[-1] ) < 2: a = a[:-1]

	a=a[:2000]

	plt.plot(
		range( len(a) ),
		list(
			map(
				lambda e: e[ind],
				a
			)
		)
	)
	plt.show()

def transform_to_csv():
	week_0_list, week_1_list, week_2_list = pickle.load(open(
		'complete_data_set_top_80.p', 'rb'
	))

	print(len(week_0_list))
	print(week_0_list[0])
	print(week_0_list[1])
	print(week_0_list[2])

	with open('components_and_trend.csv','wt') as file:

		file.write('pc1,pc2,pc3,pc4,pc5,pc6,pc7,pc8,pc9,pc10,pc11,trend\n')

		for t in itertools.chain(week_0_list, week_1_list, week_2_list):
			for pc in t[3]:
				file.write( str(pc) + ',' )
			file.write( str(t[1]) + '\n' )

def train_main_1():
	# new_index = 283
	#	recurrent network with sigmoid activation and LeakyReLU internal

	# new_index = 302
	#	recurrent network with sigmoid activation and ReLU internal

	# new_index = 303
	#	recurrent network with sigmoid activation and ReLU internal and only
	#	one output
	# 	17000 data set

	# new_index = 304
	#	recurrent network with sigmoid activation and ReLU internal and only
	#	one output
	# 	new 14000 data set 6th may

	# new_index = 305
	#	recurrent network with sigmoid activation and ReLU internal and only
	#	one output
	# 	new 14000 data set 6th may
	#   only ars and PC1 as input

	new_index = 305

	def get_model_2(ws, bins_no):
		inp_layer = keras.layers.Input(shape=(ws, bins_no,))
		x = keras.layers.TimeDistributed(
			keras.layers.Dense(
				units=bins_no,
				activation='relu'
			)
		)(inp_layer)
		x = keras.layers.BatchNormalization()(x)
		# x = keras.layers.Dropout(0.01)(x)

		x = keras.layers.Bidirectional(
			keras.layers.LSTM(
				units=bins_no,
				return_sequences=True,
			)
		)(x)
		x = keras.layers.BatchNormalization()(x)

		x = keras.layers.TimeDistributed(
			keras.layers.Dense(units=10,
					activation='relu')
		)(x)
		x = keras.layers.BatchNormalization()(x)

		x = keras.layers.TimeDistributed(
			keras.layers.Dense(units=5,
					activation='relu')
		)(x)
		x = keras.layers.BatchNormalization()(x)

		x = keras.layers.Bidirectional(
			keras.layers.LSTM(
				units=1,
				return_sequences=False,
			)
		)(x)
		x = keras.layers.BatchNormalization()(x)

		# x = keras.layers.TimeDistributed(
		# 	keras.layers.Dense(units=1, activation='sigmoid')
		# )(x)

		x = keras.layers.Dense(units=1, activation='sigmoid')(x)

		return keras.models.Model(inputs=inp_layer, outputs=x)

	def get_model_3(ws, bins_no):
		inp_layer = keras.layers.Input(shape=(ws, bins_no,))
		x = keras.layers.TimeDistributed(
			keras.layers.Dense(
				units=bins_no,
				activation='relu'
			)
		)(inp_layer)
		x = keras.layers.BatchNormalization()(x)
		# x = keras.layers.Dropout(0.01)(x)

		x = keras.layers.Bidirectional(
			keras.layers.LSTM(
				units=bins_no,
				return_sequences=False,
			)
		)(x)
		x = keras.layers.BatchNormalization()(x)

		x = keras.layers.Dense(units=1, activation='sigmoid')(x)

		return keras.models.Model(inputs=inp_layer, outputs=x)

	if False:
		data_set = pickle.load(open('./pca_data_sets/5.p','rb'))[2]
	if True:
		data_set = pickle.load(open('./pca_data_sets/6.p','rb'))['train_valid_data_sets']

	if True:
		data_set =\
		(
			data_set[0],
			transform_array_only_last(data_set[1]),
			data_set[2],
			transform_array_only_last(data_set[3]),
		)
	if False:
		data_set =\
		(
			data_set[0][:,:,1:],
			data_set[1],
			data_set[2][:,:,1:],
			data_set[3],
		)
	if True:
		data_set =\
		(
			data_set[0][:,:,:4],
			data_set[1],
			data_set[2][:,:,:4],
			data_set[3],
		)

	model = get_model_3(
		40,
		4,
	)

	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	if False:
		if False:
			old_model_path = './pca_multiple_model_folders/models_283/model_0488.hdf5'
		if False:
			old_model_path = './pca_multiple_model_folders/models_303/model_0672.hdf5'
		if True:
			old_model_path = './pca_multiple_model_folders/models_304/model_0318.hdf5'

		old_model = keras.models.load_model(old_model_path)

		new_model_layers = list()
		for i in range( len( model.layers ) ):
			if model.layers[i].name.startswith('bidirectional')\
				or model.layers[i].name.startswith('time_distributed'):
				new_model_layers.append(i)

		j = 0
		for i in range( len( old_model.layers ) - 1):

			if old_model.layers[i].name.startswith('bidirectional')\
				or old_model.layers[i].name.startswith('time_distributed'):

				model.layers[ new_model_layers[j] ].set_weights(old_model.layers[i].get_weights())

				j += 1

		del old_model

	model.summary()

	print('\n\n\n' + str(new_index) + '\n\n\n')

	with tf.device('/gpu:1'):

		model.fit(
			x=data_set[0],
			y=data_set[1],
			batch_size=128,
			epochs=2000,
			validation_data=(\
				data_set[2],
				data_set[3],
			),
			verbose=2,
			callbacks=[
				keras.callbacks.CSVLogger(
					'./pca_csv_folder/losses_'\
					+ str( new_index ) + '.csv'
				),
				keras.callbacks.ModelCheckpoint(
					'pca_multiple_model_folders/'\
						+ 'models_' + str(new_index) + '/'\
						+ "model_{epoch:04d}.hdf5",
					monitor='val_loss',
					save_best_only=True
				),
				keras.callbacks.EarlyStopping(
					monitor='val_loss',
					patience=125,
				)
			]
		)

def test_reordering():
	bins_list = get_normalized_values_per_week('complete_data_set_top_80.p')

	x_per_week, y_per_week =\
		(
			list(
				map(
					lambda b_list:
						np.array(
							list(
								map(
									lambda ex: ex[:-1],
									b_list
								)
							)
						),
					bins_list
				)
			),
			list(
				map(
					lambda b_list:
						np.array(
							list(
								map(
									lambda ex: [ ex[-1] , ],
									b_list
								)
							)
						),
					bins_list
				)
			)
		)

	print('Finished loading per week !')

	x_whole, y_whole =\
		aggregate_arrays_0(
			x_per_week,
			y_per_week,
		)

	print('Finished loading whole !')

	a, b = pickle.load(open( './pca_data_sets/4.p' , 'rb' ))[2][2:]
	x_ordered, y_ordered, indexes_list =\
		reorder_valid_in_temporal_order(
			x_whole, y_whole, a, b,
		)

	print( x_ordered.shape )

	print( len( indexes_list ) )

def get_differences_in_matrices():
	week_list_dirs =\
	tuple(
		map(
			lambda fn: os.listdir(fn),
			MATRIXES_FOLDER_LIST
		)
	)

	print(
		len(
			tuple(
				filter(
					lambda e: WEEK_TIME_MOMENTS[0][0] <= e < WEEK_TIME_MOMENTS[0][1],
					map(
						lambda fn: int( fn.split('_')[0] ),
						week_list_dirs[0]
					)
				)
			)
		)
	)
	print(
		len(
			tuple(
				filter(
					lambda e: WEEK_TIME_MOMENTS[1][0] <= e < WEEK_TIME_MOMENTS[1][1],
					map(
						lambda fn: int( fn.split('_')[0] ),
						week_list_dirs[1]
					)
				)
			)
		)
	)
	print(
		len(
			tuple(
				filter(
					lambda e: WEEK_TIME_MOMENTS[2][0] <= e < WEEK_TIME_MOMENTS[2][1],
					map(
						lambda fn: int( fn.split('_')[0] ),
						week_list_dirs[2]
					)
				)
			)
		)
	)

	def get_time_tags_f(keyword):

		a_list = list()

		for ind, week_list in enumerate( week_list_dirs ):

			a_list.append(
				list(
					filter(
						lambda time_tag:\
							WEEK_TIME_MOMENTS[ ind ][0] <= time_tag < WEEK_TIME_MOMENTS[ ind ][1],
						map(
							lambda fn: int( fn.split('_')[0] ),
							filter(
								lambda fn: keyword in fn,
								week_list
							)
						)
					)
				)
			)

		return a_list

	get_diffs_f = lambda iterable:\
	tuple(
		map(
			lambda week_iterable:\
				list(map(
					lambda p: p[1] - p[0],
					zip(
						iter(tuple(week_iterable)[:-1]),
						iter(tuple(week_iterable)[1:]),
					)
				)),
			iterable
		)
	)

	distance_time_tag_list = get_diffs_f( get_time_tags_f( 'distance' ) )
	demotion_time_tag_list = get_diffs_f( get_time_tags_f( 'demotion' ) )

	print_stats_f = lambda keyword, a_iterable:\
		print(
			keyword + ': min=' + str(min(a_iterable))\
			+' avg=' + str( sum(a_iterable) / len(a_iterable) )\
			+' max=' + str( max(a_iterable) )
		)

	print_stats_f(
		'distance',
		tuple(itertools.chain.from_iterable( distance_time_tag_list ))
	)
	print_stats_f(
		'demotion',
		tuple(itertools.chain.from_iterable( demotion_time_tag_list ))
	)

	pickle.dump(
		( distance_time_tag_list , demotion_time_tag_list ),
		open(
			'pipe.p',
			'wb',
		)
	)

if __name__ == '__main__':
	global n_proc

	n_proc = 95

	# dump_nn_results(
	# 	(
	# 		'complete_data_set_top_80.p',
	# 		'/cpu:0',
	# 		( 206 , 254 )
	# 	)
	# )
	# dump_nn_results('complete_data_set_top_80.p', '/gpu:3', ( 36 , 45 ) )
	# Pool(4).map(
	# 	dump_nn_results,
	# 	(
	# 		( 'complete_data_set_top_80.p', '/gpu:0', ( 206 , 219 ) ),
	# 		( 'complete_data_set_top_80.p', '/gpu:1', ( 219 , 232 ) ),
	# 		( 'complete_data_set_top_80.p', '/gpu:2', ( 232 , 254 ) ),
	# 	)
	# )

	# dump_pca_info_for_plot()

	# test_generator()
	# train_main_0('complete_data_set_top_80.p')
	# get_results_0(
	# 	'complete_data_set_top_80.p',
	# 	'pca_multiple_model_folders/models_13/model_6455.hdf5'
	# )

	# hyper_parameters_sweep_train()

	# train_main()

	# train_main_1()

	dump_results_main()

	# test_reordering()

	# analyse_data_set()

	# get_differences_in_matrices()

	if False:
		with open('pca_what_is_what.txt', 'wt') as myfile:
			for _, content in sorted( map(
				lambda fn: ( int(fn[:-4]) ,  open('pca_index_meaning/'+fn).read() ,),
				os.listdir( 'pca_index_meaning' ) ) ):
					myfile.write(content + '\n')
