import csv
import itertools
import pickle
import random
import numpy as np

PCA_DUMP_FOLDER='./pca_dumps/'

def get_thp(thp_file, time_interval):
	'''
	Generates a list of (time_tag, trend_value) using the moving averages window size
	that gives the closest kurtosis to 0.
	'''
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
					thp_list,
				)
			)
		)
	)

def get_thp_1(thp_file, time_interval):
	'''
	Generates a list of (time_tag, trend_value) using the moving averages window size
	that gives the closest kurtosis to 0.
	'''

	print('Entered',thp_file)

	from statsmodels.tsa.seasonal import seasonal_decompose

	thp_gen = csv.reader(open(thp_file,'rt'))

	next(thp_gen)

	thp_list = list()

	for line in thp_gen:

		# print(len(line),len(line[-1]))

		if len(line) >= 2 and len(line[-1]) > 2:

			thp_list.append((\
				1000 * int(line[0]),
				float(line[-1]),
			))

	new_thp_list = list()
	for prev_t, next_t in zip( thp_list[:-1] , thp_list[1:] ):

		new_thp_list.append(prev_t)

		time_diff = next_t[0] - prev_t[0]

		if time_diff > 120000:

			t = prev_t[0] + 120000

			while t < next_t[0]:

				new_thp_list.append(
					(
						t,
						prev_t[1],
					)
				)

				t += 120000

	print(len(thp_list))

	new_thp_list.append( thp_list[-1] )

	thp_list = new_thp_list

	for prev_t, next_t in zip( thp_list[:-1] , thp_list[1:] ):
		if next_t[0] - prev_t[0] != 120000:
			print('ii diferit !!!')

	return\
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
					thp_list,
				)
			)
		)
	)

def get_training_set(thp_iterable, read_size_iterable, matrices_iterable):
	'''
	Does the matching for the trend, average read size and whole matrices based on
	time tags.
	'''
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

def get_dist_dict_by_time(time_moment,matrices_folder):
	'''
	This is used to parse a raw distance matrix file into a Python dictionary
	object.
	'''
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
	'''
	This is used to parse a raw demotion matrix file into a Python dictionary
	object.
	'''
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

def create_data_set(thp_iterable, read_size_iterable, distance_iterable, demotion_iterable, ignore_read_size_flag=False):
	'''
	Matches the trend, average read size, distance and demotion matrices based on time tags.
	'''
	def reduce_throughput_at_front(thp_iterable_0, a_iterable, time_margin=0):
		thp_i = 0

		while thp_iterable_0[thp_i][0] - time_margin < a_iterable[0][0]: thp_i += 1

		return thp_iterable_0[thp_i:]

	if not ignore_read_size_flag:
		thp_iterable = reduce_throughput_at_front(thp_iterable, read_size_iterable, 120000)

	thp_iterable = reduce_throughput_at_front(thp_iterable, distance_iterable)

	thp_iterable = reduce_throughput_at_front(thp_iterable, demotion_iterable)

	def reduce_throughput_at_back(thp_iterable_0, a_iterable):
		thp_i = len( thp_iterable_0 ) - 1

		while thp_iterable_0[thp_i][0] > a_iterable[-1][0]: thp_i -= 1

		return thp_iterable_0[:thp_i + 1]

	if not ignore_read_size_flag:
		thp_iterable = reduce_throughput_at_back(thp_iterable, read_size_iterable)

	thp_iterable = reduce_throughput_at_back(thp_iterable, distance_iterable)

	thp_iterable = reduce_throughput_at_back(thp_iterable, demotion_iterable)

	data_set_list = list()

	thp_i = 0

	if not ignore_read_size_flag:
		fi_rs_i = 0
		while read_size_iterable[fi_rs_i][0] < thp_iterable[thp_i][0] - 120000:
			fi_rs_i+=1

		time_window_rs = 0
		la_rs_i = fi_rs_i
		while True:
			time_window_rs += read_size_iterable[la_rs_i][1]
			la_rs_i += 1
			if read_size_iterable[la_rs_i][0] > thp_iterable[thp_i][0]:
				break

	dist_i = dem_i = 0

	while thp_i < len(thp_iterable):

		if not ignore_read_size_flag:
			while read_size_iterable[fi_rs_i][0] < thp_iterable[thp_i][0] - 120000:
				time_window_rs -= read_size_iterable[fi_rs_i][1]
				fi_rs_i+=1

			if read_size_iterable[la_rs_i][0] <= thp_iterable[thp_i][0]:
				while True:
					time_window_rs += read_size_iterable[la_rs_i][1]
					la_rs_i += 1
					if read_size_iterable[la_rs_i][0] > thp_iterable[thp_i][0]:
						break

		while dist_i < len(distance_iterable) - 1\
			and distance_iterable[dist_i+1][0] <= thp_iterable[thp_i][0]:
			dist_i += 1

		while dem_i < len(demotion_iterable) - 1\
			and demotion_iterable[dem_i+1][0] <= thp_iterable[thp_i][0]:
			dem_i += 1

		data_set_list.append(
			(
				thp_iterable[thp_i][0],
				thp_iterable[thp_i][1],
				( time_window_rs / 120 ) if not ignore_read_size_flag else 0,
				distance_iterable[dist_i][1],
				demotion_iterable[dem_i][1],
			)
		)

		thp_i += 1

	return data_set_list

def get_clients_and_ses_minimal_list_1(whole_matrix_per_week_list):
	'''
	Creates the minimal data set.
	'''
	clients_sorted_list = sorted( whole_matrix_per_week_list[0][0].keys() )

	for se_dict in whole_matrix_per_week_list[0][0].values():

		se_sorted_list = sorted( se_dict.keys() )

		break

	for distance_list in whole_matrix_per_week_list:
		for cl_dict in distance_list:

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

	return clients_sorted_list, se_sorted_list

def get_normalized_values_per_week(data_set_file):
	# week_0_list, week_1_list, week_2_list = pickle.load(open(\
	# 	PCA_DUMP_FOLDER + data_set_file, 'rb'
	# ))
	week_list = pickle.load(open(\
		PCA_DUMP_FOLDER + data_set_file, 'rb'
	))

	min_thp = max_thp = week_list[0][0][1]

	min_rs = max_rs = week_list[0][0][2]

	min_comp = min(week_list[0][0][3])

	max_comp = max(week_list[0][0][3])

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

	for _ , thp_val, avg_rs, pca_comp_list in itertools.chain.from_iterable(week_list):
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
	return\
		tuple(
			map(
				lambda e_week: process_data_list( e_week ),
				week_list,
			)
		)

def normalize_and_split_data_set_1(data_set_file,time_window,max_overlap_length,return_indexes_flag=False):

	bins_list = get_normalized_values_per_week(data_set_file)

	# print(\
	# 	'bins_list normalization: '\
	# 	+ str(len(bins_list)) + ' '\
	# 	+ str(len(bins_list[0])) + ' '\
	# 	+ str(len(bins_list[1])) + ' '\
	# )

	new_bin_list = list()

	i = 0
	for b_list in bins_list:

		for j in range(len(b_list)-time_window):

			new_bin_list.append(list())

			for l in range( j , j + time_window , 1 ):

				new_bin_list[-1].append(list())

				for e in b_list[ l ]:
					new_bin_list[-1][-1].append(e)

			i+=1

	bins_list = new_bin_list

	print('bins_list after time seq transform: '\
		+ str(len(bins_list)) + ' '\
		+ str(len(bins_list[0])) + ' '\
		+ str(len(bins_list[0][0])) + ' '\
	)

	new_bin_list = list()

	last_appended_example_index = 0

	new_bin_list.append(bins_list[0])

	for ind , example in zip(
		range(1,len(bins_list)),
		bins_list[1:]):

		if time_window - ind + last_appended_example_index <= max_overlap_length:

			last_appended_example_index = ind

			new_bin_list.append( example )

	bins_list = new_bin_list

	valid_indexes_list = random.sample(
		range( len( bins_list ) ),
		round( 0.2 * len( bins_list ) ),
	)
	train_indexes_list = tuple(filter(
		lambda i: i not in valid_indexes_list,range(len( bins_list ))))

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

	if return_indexes_flag:
		return\
			(\
				train_indexes_list,\
				valid_indexes_list,\
			),\
			(\
				np.array(train_x_list),\
				np.array(train_y_list),\
				np.array(valid_x_list),\
				np.array(valid_y_list),\
			)

	return\
		np.array(train_x_list),\
		np.array(train_y_list),\
		np.array(valid_x_list),\
		np.array(valid_y_list)

def transform_array_only_last(arr):
	'''
	Transforms a time sequence y data set with all units output into one
	where only the last unit outputs something.
	'''
	new_arr = np.empty( ( arr.shape[0] , 1 ) )
	for i in range( arr.shape[0] ): new_arr[i,0] = arr[i,-1,0]
	return new_arr

def generate_random_batch_1(batch_size, window_size, indexes_list, data_set_dict,\
	only_last_flag=False, firs_bs_elements_flag=False, log_freq=None, one_input_flag=False):
	'''
	Returns a batch that is ready to be fed into the encoder-decoder architecture.

	batch_size
		Batch size to be used when training

	window_size
		The size of the time sequence used (usually 40)

	indexes_list
		An iterable consisting of indexes from which the batch is chosen

	data_set_dict
		A dictionary which must hold a 'non_split_data_set' tag containing the non-time-sequence
		data set

	only_last_flag
		Boolenea on wether to create a data set where only the last output of a
		recurrent unit should be taken into account or not

	firs_bs_elements_flag
		Boolean on wether to return the first batch_size elements in the data set
		or not

	log_freq
		Debug boolean

	one_input_flag
		Boolean on wether the encoder should downsize the average read size along with
		the big matrix or not
	'''
	if one_input_flag:
		x1_list = np.empty( (\
			batch_size,\
			window_size,\
			len( data_set_dict['non_split_data_set'][0] ) - 1\
		) )
		y1_list = np.empty( (\
			batch_size,\
			window_size,\
			len( data_set_dict['non_split_data_set'][0] ) - 1\
		) )
	else:
		x1_list = np.empty( ( batch_size , window_size , 1 ) )
		x2_list = np.empty( (\
			batch_size,\
			window_size,\
			len( data_set_dict['non_split_data_set'][0] ) - 2\
		) )
		y1_list = np.empty( (\
			batch_size,\
			window_size,\
			len( data_set_dict['non_split_data_set'][0] ) - 2\
		) )

	if only_last_flag:
		y2_list = np.empty( ( batch_size , 1 , ) )
	else:
		y2_list = np.empty( ( batch_size , window_size , 1 , ) )

	if firs_bs_elements_flag:
		chosen_indexes_list = indexes_list[:batch_size]
	else:
		chosen_indexes_list = random.sample( indexes_list , batch_size )

	for i_batch, i_random in\
		enumerate( chosen_indexes_list ):

		if one_input_flag:
			x1_list[ i_batch , : , : ] = y1_list[ i_batch , : , : ] = data_set_dict['non_split_data_set'][\
				i_random-window_size:i_random, :-1]
		else:
			x1_list[ i_batch , : , 0 ] = data_set_dict['non_split_data_set'][\
				i_random-window_size:i_random, 0]
			x2_list[ i_batch , : , : ] = y1_list[ i_batch , : , : ] = data_set_dict['non_split_data_set'][\
				i_random-window_size:i_random, 1:-1]

		if only_last_flag:
			y2_list[i_batch,0] = data_set_dict['non_split_data_set'][i_random-1,-1]
		else:
			y2_list[ i_batch , : , 0 ] = data_set_dict['non_split_data_set'][\
				i_random-window_size:i_random, -1]

		if log_freq is not None:
			if i_batch % 1000 == 0:
				print('Got to', i_batch)

	if one_input_flag:
		return x1_list , [ y1_list , y2_list , ]
	return\
		[ x1_list , x2_list , ] , [ y1_list , y2_list , ]
