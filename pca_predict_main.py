import os
import matplotlib.pyplot as plt
from shutil import copyfile
import pickle
import itertools
from sklearn.decomposition import PCA
import numpy as np
from multiprocessing import Pool
import json
import csv
import random
# import keras
from tensorflow import keras

MATRIXES_FOLDER_LIST = (\
	'./remote_host_0/log_folder/',\
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

	def get_dist_dict_by_time(time_moment,matrices_folder):
		d = dict()
		for cl, se, val in\
			map(
				lambda e: ( e[0].lower() , ( e[1][0].lower() , e[1][1].lower() , ) , e[2] , ),
				map(
					lambda e: (e[0], e[1].split('::')[1:], float(e[2]),),
					map(
						lambda r: r.split(';'),
						open(matrices_folder + str(time_moment) + '_distance','r').read().split('\n')[1:-1]
					)
				)
			):
			if cl in d:
				d[cl][se] = val
			else:
				d[cl] = { se : val }
		return d

	def get_dem_dict_by_time(time_moment,matrices_folder):
		d = dict()
		for se, val in\
			map(
				lambda e: ( ( e[0][0].lower() , e[0][1].lower() , ) , e[1] , ),
				map(
					lambda e: (e[0].split('::')[1:], float(e[3]),),
					map(
						lambda r: r.split(';'),
						open(matrices_folder + str(time_moment) + '_demotion','r').read().split('\n')[1:-1]
					)
				)
			):
			d[se] = val
		return d

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

	def get_thp(thp_file, time_interval):
		from statsmodels.tsa.seasonal import seasonal_decompose

		thp_gen = csv.reader(open(thp_file))

		next(thp_gen)

		thp_list = list()

		for line in thp_gen:
			if len(line) == 2:

				thp_list.append((\
					1000 * int(line[0]),
					float(line[1]),
				))

		return \
		list(
			map(
				lambda p: ( p[1][0] , p[0] , ) ,
				filter(
					lambda p: str(p[0]) != 'nan'\
						and time_interval[0] <= p[1][0] < time_interval[1],
					zip(
						seasonal_decompose(
							tuple(
								map(
									lambda e: e[1],
									thp_list
								)
							),
							model='additive',
							freq=4260
						).trend,
						thp_list
					)
				)
			)
		)

	def get_training_set(thp_iterable, read_size_iterable, matrices_iterable):
		thp_i = 0
		time_window_rs = 0

		data_set_list = list()

		while thp_iterable[thp_i][0] - 120000 < read_size_iterable[0][0]: thp_i += 1

		while thp_iterable[thp_i][0] < matrices_iterable[0][0]: thp_i += 1

		clip_i = len(thp_iterable) - 1
		while thp_iterable[clip_i][0] > read_size_iterable[-1][0]: clip_i -= 1
		thp_iterable = thp_iterable[:clip_i + 1]

		clip_i = len(thp_iterable) - 1
		while thp_iterable[clip_i][0] > matrices_iterable[-1][0]: clip_i -= 1
		thp_iterable = thp_iterable[:clip_i + 1]

		del clip_i

		fi_rs_i = 0
		while read_size_iterable[fi_rs_i][0] < thp_iterable[thp_i][0] - 120000:
			fi_rs_i+=1

		la_rs_i = fi_rs_i
		while True:
			time_window_rs += read_size_iterable[la_rs_i][1]
			la_rs_i += 1
			if read_size_iterable[la_rs_i][0] > thp_iterable[thp_i][0]:
				break

		mat_i = 0
		while mat_i < len(matrices_iterable) - 1\
			and matrices_iterable[mat_i+1][0] <= thp_iterable[thp_i][0]:
			mat_i += 1

		while thp_i < len(thp_iterable):

			while read_size_iterable[fi_rs_i][0] < thp_iterable[thp_i][0] - 120000:
				time_window_rs -= read_size_iterable[fi_rs_i][1]
				fi_rs_i+=1

			if read_size_iterable[la_rs_i][0] <= thp_iterable[thp_i][0]:
				while True:
					time_window_rs += read_size_iterable[la_rs_i][1]
					la_rs_i += 1
					if read_size_iterable[la_rs_i][0] > thp_iterable[thp_i][0]:
						break

			while mat_i < len(matrices_iterable) - 1\
				and matrices_iterable[mat_i+1][0] <= thp_iterable[thp_i][0]:
				mat_i += 1

			data_set_list.append(
				(
					thp_iterable[thp_i][0],
					thp_iterable[thp_i][1],
					time_window_rs / 120,
					matrices_iterable[mat_i][1],
				)
			)

			thp_i += 1

		return data_set_list

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
	week_0_list, week_1_list, week_2_list = pickle.load(open(
		PCA_DUMP_FOLDER + data_set_file, 'rb'
	))

	min_thp = max_thp = week_0_list[0][1]

	min_rs = max_rs = week_0_list[0][2]

	min_comp = min(week_0_list[0][3])

	max_comp = max(week_0_list[0][3])

	def get_min_max(value, min_a, max_a, is_iterable_flag=False):
		if is_iterable_flag:
			a, b = min(value), max(value)
			return\
				(
					a if a < min_a else min_a,
					b if b > max_a else max_a,
				)
		return (
			value if value < min_a else min_a,
			value if value > max_a else max_a,
		)

	for _ , thp_val, avg_rs, pca_comp_list in itertools.chain(week_0_list,week_1_list,week_2_list):
		min_thp, max_thp = get_min_max(thp_val, min_thp, max_thp,)
		min_rs, max_rs = get_min_max(avg_rs, min_rs, max_rs,)
		min_comp, max_comp = get_min_max(pca_comp_list, min_comp, max_comp, True)

	process_data_list=lambda l:\
	list(
		map(
			lambda p:\
				[2*(p[2]-min_rs)/(max_rs-min_rs)-1,]\
				+ list(
					map(
						lambda e: 2*(e-min_comp)/(max_comp-min_comp)-1,
						p[3]
					)
				)\
				+ [(p[1]-min_thp)/(max_thp-min_thp),],
			l
		)
	)
	# process_data_list=lambda l:\
	# list(
	# 	map(
	# 		lambda p:\
	# 			list(
	# 				map(
	# 					lambda e: 2*(e-min_comp)/(max_comp-min_comp)-1,
	# 					p[3]
	# 				)
	# 			)\
	# 			+ [(p[1]-min_thp)/(max_thp-min_thp),],
	# 		l
	# 	)
	# )


	bins_list =\
		(process_data_list(week_0_list),\
		process_data_list(week_1_list),\
		process_data_list(week_2_list))

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

	if True:
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

	if False:
		return np.array(train_x_list),\
			np.array(train_y_list),\
			np.array(valid_x_list),\
			np.array(valid_y_list)

def train_main():
	def get_model_1(ws, bins_no):
		inp_layer = keras.layers.Input(shape=(ws, bins_no,))

		x = inp_layer

		# x = keras.layers.TimeDistributed(
		# 	keras.layers.Dense(units=1000)
		# )(inp_layer)
		# x = keras.layers.LeakyReLU(alpha=0.3)(x)
		# x = keras.layers.Dropout(0.1)(x)

		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Bidirectional(
			keras.layers.LSTM(
				units=bins_no,
				return_sequences=True,
			)
		)(x)
		# x = keras.layers.Dropout(0.05)(x)

		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.TimeDistributed(
			keras.layers.Dense(units=10,)
		)(x)
		x = keras.layers.LeakyReLU(alpha=0.3)(x)

		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.TimeDistributed(
			keras.layers.Dense(units=5,)
		)(x)
		x = keras.layers.LeakyReLU(alpha=0.3)(x)

		# x = keras.layers.BatchNormalization()(x)
		# x = keras.layers.TimeDistributed(
		# 	keras.layers.Dense(units=100,)
		# )(x)
		# x = keras.layers.LeakyReLU(alpha=0.3)(x)

		# x = keras.layers.BatchNormalization()(x)
		# x = keras.layers.TimeDistributed(
		# 	keras.layers.Dense(units=25,)
		# )(x)
		# x = keras.layers.LeakyReLU(alpha=0.3)(x)

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

	model = get_model_1(X_train.shape[1], X_train.shape[2])

	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	model.summary()

	model.fit(
		x=X_train,
		y=y_train,
		batch_size=32,
		epochs=100000,
		validation_data=(\
			X_valid,
			y_valid,
		),
		callbacks=[
			keras.callbacks.ModelCheckpoint(out_folder + "model_{epoch:04d}.hdf5", monitor='loss', save_best_only=True),
		]
	)

def analyse_data_set():
	week_0_list, week_1_list, week_2_list = pickle.load(open(
		'complete_data_set_top_80.p', 'rb'
	))

	min_thp = max_thp = week_0_list[0][1]

	min_rs = max_rs = week_0_list[0][2]

	min_comp = min(week_0_list[0][3])

	max_comp = max(week_0_list[0][3])

	def get_min_max(value, min_a, max_a, is_iterable_flag=False):
		if is_iterable_flag:
			a, b = min(value), max(value)
			return\
				(
					a if a < min_a else min_a,
					b if b > max_a else max_a,
				)
		return (
			value if value < min_a else min_a,
			value if value > max_a else max_a,
		)

	for _ , thp_val, avg_rs, pca_comp_list in itertools.chain(week_0_list,week_1_list,week_2_list):
		min_thp, max_thp = get_min_max(thp_val, min_thp, max_thp,)
		min_rs, max_rs = get_min_max(avg_rs, min_rs, max_rs,)
		min_comp, max_comp = get_min_max(pca_comp_list, min_comp, max_comp, True)

	process_data_list=lambda l:\
	list(
		map(
			lambda p:\
				[2*(p[2]-min_rs)/(max_rs-min_rs)-1,]\
				+ list(
					map(
						lambda e: 2*(e-min_comp)/(max_comp-min_comp)-1,
						p[3]
					)
				)\
				+ [2*(p[1]-min_thp)/(max_thp-min_thp)-1,],
			l
		)
	)

	bins_list =\
		(process_data_list(week_0_list),\
		process_data_list(week_1_list),\
		process_data_list(week_2_list))

	plt.plot(
		range(len(bins_list[0]) + len(bins_list[1]) + len(bins_list[2])),
		list(map(lambda e: e[-1], bins_list[0] + bins_list[1] + bins_list[2]))
	)

	plt.plot(
		range(len(bins_list[0]) + len(bins_list[1]) + len(bins_list[2])),
		list(map(lambda e: e[1], bins_list[0] + bins_list[1] + bins_list[2]))
	)

	plt.show()

def analyse_data_set_1():
	x_train, y_train, x_valid, y_valid = json.load(
		open(
			'ready_to_train_data_set.json',
			'rt'
		)
	)

	plt.plot(
		range( len(x_train) ),
		list(
			map(
				lambda a: a[2][5],
				x_train
			)
		)
	)

	plt.plot(
		range( len(y_train) ),
		list(
			map(
				lambda a: 2 * a[2][0] - 1,
				y_train
			)
		)
	)

	plt.show()

if __name__ == '__main__':
	global n_proc

	n_proc = 95

	analyse_data_set_1()
