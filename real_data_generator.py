import csv
import os
from multiprocessing import Pool, Manager
import pickle
from functools import reduce
import pickle
import json
# import numpy as np
# from scipy import interpolate
from functools import reduce
import random
import matplotlib.pyplot as plt
# from statsmodels.tsa.seasonal import seasonal_decompose

def generator_iterate(iterable):
	'''
	Helps with iteration over a generator.
	'''
	try:
		return next(iterable)
	except StopIteration:
		return None

def get_matrices_time_moments_lists(first_moment, last_moment):
	'''
	Selects the matrices which are temporally located between the 2 moments.
	'''
	filename_tuple = tuple(os.listdir('./remote_host/log_folder/'))
	distance_list = list(
		map(
			lambda e: int(e.split('_')[0]),
			filter(\
				lambda fn: '_distance' in fn,
				filename_tuple
			)
		)
	)
	demotion_list = list(
		map(
			lambda e: int(e.split('_')[0]),
			filter(\
				lambda fn: '_demotion' in fn,
				filename_tuple
			)
		)
	)
	distance_list.sort()
	demotion_list.sort()

	i = len(distance_list)-1
	while distance_list[i] > first_moment: i-=1
	distance_list = distance_list[i:]

	i = len(demotion_list)-1
	while demotion_list[i] > first_moment: i-=1
	demotion_list = demotion_list[i:]

	i = 0
	while i < len(distance_list) and distance_list[i] < last_moment: i+=1
	distance_list = distance_list[:i+1]

	i = 0
	while i < len(demotion_list) and demotion_list[i] < last_moment: i+=1
	demotion_list = demotion_list[:i+1]

	return distance_list, demotion_list

def get_dist_dict_by_time(time_moment):
	'''
	Gets a UNIX time moment. Read the distance CSV file associated with it and parse it.
	Returns a dict: { client : { storage element : distance value } }.
	'''
	d = dict()
	for cl, se, val in\
		map(
			lambda e: ( e[0].lower() , ( e[1][0].lower() , e[1][1].lower() , ) , e[2] , ),
			map(
				lambda e: (e[0], e[1].split('::')[1:], float(e[2]),),
				map(
					lambda r: r.split(';'),
					open('./remote_host/log_folder/' + str(time_moment) + '_distance','r').read().split('\n')[1:-1]
				)
			)
		):
		if cl in d:
			d[cl][se] = val
		else:
			d[cl] = { se : val }
	return d

def get_dem_dict_by_time(time_moment):
	'''
	Gets a UNIX time moment. Read the demotion CSV file associated with it and parse it.
	Returns a dict: { storage element : distance value }.
	'''
	d = dict()
	for se, val in\
		map(
			lambda e: ( ( e[0][0].lower() , e[0][1].lower() , ) , e[1] , ),
			map(
				lambda e: (e[0].split('::')[1:], float(e[3]),),
				map(
					lambda r: r.split(';'),
					open('./remote_host/log_folder/' + str(time_moment) + '_demotion','r').read().split('\n')[1:-1]
				)
			)
		):
		d[se] = val
	return d

def get_matrices(first_moment, last_moment):
	'''
	Return pairs of time and associate dictionary for the demotion and distance matrices.
	'''
	distance_list, demotion_list = get_matrices_time_moments_lists(first_moment, last_moment)

	dist_res_list = []
	for time_moment in distance_list:
		dist_res_list.append((time_moment,get_dist_dict_by_time(time_moment),))

	dem_res_list = []
	for time_moment in demotion_list:
		dem_res_list.append((time_moment,get_dem_dict_by_time(time_moment),))

	return dist_res_list, dem_res_list

def answer_per_proc_0(i):
	ref_dict = get_dist_dict_by_time(g_dist_list[-1])
	for k in ref_dict.keys():
		if g_queries_list[i][1] in k:
			client_name = k
			break

	if g_queries_list[i][0] >= g_dist_list[-1]:
		local_dist_dict = ref_dict[client_name]
	else:
		for j in range(len(g_dist_list)-1):
			if g_dist_list[j] <= g_queries_list[i][0] < g_dist_list[j+1]:
				local_dist_dict = get_dist_dict_by_time(g_dist_list[j])[client_name]
				break

	if g_queries_list[i][0] >= g_dem_list[-1]:
		local_dem_dict = get_dem_dict_by_time(g_dem_list[-1])
	else:
		for j in range(len(g_dem_list)-1):
			if g_dem_list[j] <= g_queries_list[i][0] < g_dem_list[j+1]:
				local_dem_dict = get_dem_dict_by_time(g_dem_list[j])
				break

	se_list = []
	j = 0
	for se in g_queries_list[i][2]:
		se_list.append((
			local_dist_dict[se] + local_dem_dict[se],
			j
		))
		j+=1
	se_list.sort(key=lambda p: p[0])

	return tuple(map(lambda p: p[1], se_list))

def answer_queries_0(first_moment, last_moment,):
	global g_queries_list, g_dist_list, g_dem_list
	g_queries_list = list(pickle.load(open('unanswered_cern_queries.p','rb')))
	g_dist_list, g_dem_list = get_matrices_time_moments_lists(first_moment, last_moment)

	aaa = 0

	i = 0
	for order_list in Pool(n_proc).map(answer_per_proc_0, range(len(g_queries_list))):

		if aaa < 100:
			print(g_queries_list[i])

		g_queries_list[i] = (
			g_queries_list[i][0],
			g_queries_list[i][1],
			tuple(
				g_queries_list[i][2][order_list[j]] for j in range(len(order_list))
			),
			g_queries_list[i][3],
		)

		if aaa < 100:
			print(g_queries_list[i])
			print()

		i+=1

		aaa+=1

	pickle.dump(
		g_queries_list,
		open('answered_cern_queries.p', 'wb')
	)

def answer_per_proc_1(i):
	ref_dict = g_dist_list[-1][1]
	for k in ref_dict.keys():
		if g_queries_list[i][1] in k:
			client_name = k
			break

	if g_queries_list[i][0] >= g_dist_list[-1][0]:
		local_dist_dict = ref_dict[client_name]
	else:
		for j in range(len(g_dist_list)-1):
			if g_dist_list[j][0] <= g_queries_list[i][0] < g_dist_list[j+1][0]:
				local_dist_dict = g_dist_list[j][1][client_name]
				break

	if g_queries_list[i][0] >= g_dem_list[-1][0]:
		local_dem_dict = g_dem_list[-1][1]
	else:
		for j in range(len(g_dem_list)-1):
			if g_dem_list[j][0] <= g_queries_list[i][0] < g_dem_list[j+1][0]:
				local_dem_dict = g_dem_list[j][1]
				break

	se_list = []
	j = 0
	for se in g_queries_list[i][2]:
		se_list.append((
			local_dist_dict[se] + local_dem_dict[se],
			j
		))
		j+=1
	se_list.sort(key=lambda p: p[0])

	return tuple(map(lambda p: p[1], se_list))

def answer_queries_1(first_moment, last_moment, query_window=100000):
	global g_dist_list, g_dem_list, g_queries_list

	number_of_queries = len(pickle.load(open('unanswered_cern_queries.p','rb')))

	print('queries to process: ' + str(number_of_queries))

	g_dist_list, g_dem_list = get_matrices(first_moment, last_moment)

	print('Got matrices !')

	answered_query_fn_list = []

	a = 0
	b = 0

	while a < number_of_queries:

		g_queries_list = pickle.load(open('unanswered_cern_queries.p','rb'))[a:\
			((a + query_window) if a + query_window <= number_of_queries else number_of_queries)\
		]

		p = Pool(n_proc)

		i = 0
		for order_list in p.map(answer_per_proc_1, range(len(g_queries_list))):

			g_queries_list[i] = (
				g_queries_list[i][0],
				g_queries_list[i][1],
				tuple(
					g_queries_list[i][2][order_list[j]] for j in range(len(order_list))
				),
				g_queries_list[i][3],
			)

			i += 1

		p.close()

		answered_query_fn_list.append(
			'./p/' + str(b) + '.p'
		)

		pickle.dump(
			g_queries_list,
			open(answered_query_fn_list[-1], 'wb')
		)

		print('Finished from: ' + str(a))

		a += query_window
		b += 1

	del g_dist_list
	del g_dem_list
	del g_queries_list

	answered_query_list = []

	for fn in answered_query_fn_list:
		answered_query_list += pickle.load(open(fn,'rb'))

	pickle.dump(
		answered_query_list,
		open('answered_cern_queries.p', 'wb')
	)

def associate_per_proc(i):
	g_dict[g_thp_list[i]] =\
	list(
		filter(
			lambda t: g_thp_list[i-1][0] <= t[0] < g_thp_list[i][0],
			g_queries_list
		)
	)

def associate_queries(first_moment, last_moment, client_name, query_window=120000):
	global g_queries_list, g_thp_list, g_dict

	# g_thp_list = get_throughput(first_moment, last_moment, client_name)
	g_thp_list = pickle.load(open('throughput_dump.p','rb'))

	m = Manager()

	number_of_queries = len(pickle.load(open('answered_cern_queries.p','rb')))

	print('queries to process: ' + str(number_of_queries))

	answered_query_fn_list = []

	a = 0
	b = 0

	while a < number_of_queries:

		g_queries_list = pickle.load(open('answered_cern_queries.p','rb'))[a:\
			((a + query_window) if a + query_window <= number_of_queries else number_of_queries)\
		]

		g_dict = m.dict()

		p = Pool(n_proc)

		p.map(associate_per_proc, range(1, len(g_thp_list)))

		p.close()

		answered_query_fn_list.append(
			'./p/' + str(b) + '.p'
		)

		pickle.dump(
			dict( g_dict ),
			open(answered_query_fn_list[-1], 'wb')
		)

		del g_dict

		print('Finished from: ' + str(a) + '/' + str(number_of_queries))

		a += query_window
		b += 1

	del g_queries_list

	data_set_dict = dict()

	for fn in answered_query_fn_list:
		d = pickle.load(open(fn,'rb'))
		for k in d:
			if k in data_set_dict:
				data_set_dict[k] += d[k]
			else:
				data_set_dict[k] = d[k]

	pickle.dump(
		data_set_dict,
		open('queries_throughput_dict.p', 'wb')
	)

def associate_per_proc_1(i):
	g_dict[g_thp_list[i]] =\
	list(
		filter(
			lambda t: g_thp_list[i-1][0] <= t[0] < g_thp_list[i][0],
			g_queries_list[ g_st : g_fi ]
		)
	)

def associate_queries_1(first_moment, last_moment, client_name, query_window=12000):
	global g_queries_list, g_thp_list, g_dict, g_st, g_fi

	# g_thp_list = get_throughput(first_moment, last_moment, client_name)
	g_thp_list = pickle.load(open('throughput_dump.p','rb'))

	m = Manager()

	g_queries_list = tuple(
		filter(
			lambda t: 'cern' in t[2][0][0],
			pickle.load(open('answered_cern_queries.p','rb'))
		)
	)

	number_of_queries = len(g_queries_list)

	print('queries to process: ' + str(number_of_queries))

	answered_query_fn_list = []

	a = 0
	b = 0

	g_dict = m.dict()

	while a < number_of_queries:
		g_st, g_fi =\
			a, ((a + query_window) if a + query_window <= number_of_queries else number_of_queries)

		p = Pool(n_proc)

		p.map(
			associate_per_proc_1,
			range(1, len(g_thp_list))
		)

		p.close()

		answered_query_fn_list.append(
			'./p/' + str(b) + '.p'
		)

		pickle.dump(
			dict( g_dict ),
			open(answered_query_fn_list[-1], 'wb')
		)

		g_dict.clear()

		print('Finished from: ' + str(a) + '/' + str(number_of_queries))

		a += query_window
		b += 1

	del g_queries_list

	data_set_dict = dict()

	for fn in answered_query_fn_list:
		d = pickle.load(open(fn,'rb'))
		for k in d:
			if k in data_set_dict:
				data_set_dict[k] += d[k]
			else:
				data_set_dict[k] = d[k]

	pickle.dump(
		data_set_dict,
		open('queries_throughput_dict.p', 'wb')
	)

def assign_interpolated_thp():
	'''
	Assigns throughput to queries individually by querying an interpolated throughput
	function.
	'''
	thp_list = pickle.load(open('throughput_dump.p', 'rb'))

	thp_func = interpolate.interp1d(
		list(map(lambda p: p[0], thp_list)),
		list(map(lambda p: p[1], thp_list)),
	)

	del thp_list

	print('Will start assigning !')

	pickle.dump(
		sorted(
			reduce(
				lambda acc, x: acc + x,
				map(
					lambda value:\
					list(
						map(
							lambda q_list: q_list + (thp_func(q_list[0]),),
							value
						)
					),
					pickle.load(open('queries_throughput_dict.p', 'rb')).values()
				),
				list()
			)
		),
		open('queries_throughput_list.p','wb')
	)

def get_thp_per_proc_1(i):
	'''
	Used to query interpolation function in parallel.
	'''
	return g_thp_func(g_queries_list[i][0])

def assign_interpolated_thp_in_parallel():
	'''
	Assigns throughput to queries individually by querying an interpolated throughput
	function.
	'''
	global g_thp_func,g_queries_list

	thp_list = pickle.load(open('throughput_dump.p', 'rb'))

	g_thp_func = interpolate.interp1d(
		list(map(lambda p: p[0], thp_list)),
		list(map(lambda p: p[1], thp_list)),
	)

	del thp_list

	g_queries_list = reduce(
		lambda acc, x: acc + x,
		pickle.load(open('queries_throughput_dict.p', 'rb')).values(),
		list()
	)

	i = 0
	for thp in Pool(n_proc).map(get_thp_per_proc_1,range(len(g_queries_list))):
		g_queries_list[i] = g_queries_list[i] + (thp.item(0),)
		i+=1

	pickle.dump(
		g_queries_list,
		open('queries_throughput_list.p','wb')
	)

def normalize_q_t_list():
	'''
	Normalizes the queries-throughput structure.
	'''
	queries_list = pickle.load(open('queries_throughput_list.p', 'rb'))

	min_q_time, max_q_time = 9576450800000, -1

	min_read, max_read = 9576450800000, -1

	min_thp, max_thp = 9576450800000, -1

	next_se_index, next_cl_index = 0,0

	cl_dict, se_dict = dict(), dict()

	for q_list in queries_list:
		if min_q_time > q_list[0]: min_q_time = q_list[0]
		if max_q_time < q_list[0]: max_q_time = q_list[0]

		if min_read > q_list[-2]: min_read = q_list[-2]
		if max_read < q_list[-2]: max_read = q_list[-2]

		if min_thp > q_list[-1]: min_thp = q_list[-1]
		if max_thp < q_list[-1]: max_thp = q_list[-1]

		if q_list[1] not in cl_dict:
			cl_dict[q_list[1]] = next_cl_index
			next_cl_index += 1

		for se in q_list[2]:
			if se not in se_dict:
				se_dict[se] = next_se_index
				next_se_index += 1

	q_time_dif = max_q_time - min_q_time
	read_dif = max_read - min_read
	thp_dif = max_thp - min_thp

	for i in range(len(queries_list)):
		queries_list[i] = (
			2 * (queries_list[i][0] - min_q_time) / q_time_dif - 1,
			queries_list[i][1],
			queries_list[i][2],
			2 * (queries_list[i][3] - min_read) / read_dif - 1,
			(queries_list[i][4] - min_thp) / thp_dif
		)

	queries_dict = dict()

	for q_line in queries_list:
		if (q_line[0], q_line[-1]) in queries_dict:
			queries_dict[(q_line[0], q_line[4])].append(
				q_line[1:4]
			)
		else:
			queries_dict[(q_line[0], q_line[4])] = [q_line[1:4],]

	pickle.dump(
		(queries_dict, cl_dict, se_dict,),
		open('queries_dict_and_mapping.p','wb')
	)

def generate_new_data_set():
	ex_list = []

	min_s, max_s = 9576450800000, -1

	for k,v in pickle.load(open('queries_dict_and_mapping.p','rb'))[0].items():

		s = sum(map(lambda t: t[-1],v))

		if min_s > s: min_s = s
		if max_s < s: max_s = s

		ex_list.append(
			(
				k[0],
				s,
				k[1],
			)
		)

	for i in range(len(ex_list)):
		ex_list[i] = (
			ex_list[i][0],
			2 * (ex_list[i][1] - min_s) / (max_s - min_s) - 1,
			ex_list[i][2],
		)

	X = np.empty((len(ex_list),2,))

	i = 0
	for t in sorted(ex_list):
		X[i,0] = t[1]
		X[i,1] = t[2]
		i+=1

	pickle.dump(
		X,
		open('data_set_array_1.p','wb')
	)

def verify_per_proc(i):
	for q_list in g_answered_q_list:
		if q_list[0] == g_sub_q_list[i][0]\
			and q_list[1] == g_sub_q_list[i][1]\
			and q_list[3] == g_sub_q_list[i][3]\
			and len(q_list[2]) == len(g_sub_q_list[i][2]):

			is_se_list_identical_flag = True

			for se in q_list[2]:
				if se not in g_sub_q_list[i][2]:
					is_se_list_identical_flag = False
					break

			if is_se_list_identical_flag:
				for a in range(len(q_list[2])):
					if q_list[2][a] != g_sub_q_list[i][2][a]:
						g_lock.acquire()
						g_v.value = g_v.value + 1
						g_lock.release()
						print(q_list)
						print(g_sub_q_list[i])
						break

def verify_query_answering(queries_to_verify_count=1000):
	'''
	Randomly check some queries.
	'''
	global g_sub_q_list, g_answered_q_list, g_v, g_lock

	unansw_q_list = pickle.load(open('unanswered_cern_queries.p','rb'))

	length = len(unansw_q_list)

	g_sub_q_list = random.sample(
		unansw_q_list,
		queries_to_verify_count,
	)

	del unansw_q_list

	print('read unanswered queries !')

	m = Manager()

	g_v = m.value('i', 0)

	g_lock = m.Lock()

	print('first pass')
	g_answered_q_list = pickle.load(open('answered_cern_queries.p','rb'))[:length//4]
	print('will start pool !')
	p = Pool(n_proc)
	p.map(verify_per_proc, range(queries_to_verify_count))
	p.close()
	print()

	print('second pass')
	g_answered_q_list = pickle.load(open('answered_cern_queries.p','rb'))[length//4:length//2]
	print('will start pool !')
	p = Pool(n_proc)
	p.map(verify_per_proc, range(queries_to_verify_count))
	p.close()
	print()

	print('third pass')
	g_answered_q_list = pickle.load(open('answered_cern_queries.p','rb'))[length//2:3*length//4]
	print('will start pool !')
	p = Pool(n_proc)
	p.map(verify_per_proc, range(queries_to_verify_count))
	p.close()
	print()

	print('fourth pass')
	g_answered_q_list = pickle.load(open('answered_cern_queries.p','rb'))[3*length//4:]
	print('will start pool !')
	p = Pool(n_proc)
	p.map(verify_per_proc, range(queries_to_verify_count))
	p.close()
	print()

	print(g_v)

def get_thp_bin_per_proc(i):
	'''
	Old binning per proc.
	'''
	thp_bins_list = [0 for _ in range(g_bins_no)]

	for q_time, q_read_size in filter(\
		lambda e: g_thp_list[i][0] - g_s <= e[0] < g_thp_list[i][0] - g_f,
		g_queries_list):

		j = g_bins_no - 1

		t = g_thp_list[i][0] - g_f - g_bin_length_in_time

		while j >= 0:

			if q_time >= t:

				thp_bins_list[j] += q_read_size

				break

			t-=g_bin_length_in_time
			j-=1

	# g_dict[g_thp_list[i]] =\
	# list(
	# 	map(
	# 		lambda e: e * 1000 / g_bin_length_in_time,
	# 		thp_bins_list
	# 	)
	# )

	g_dict[g_thp_list[i]] = thp_bins_list

def get_five_minute_binned_dataset(
	first_moment,
	millis_interval_start=4000000,
	millis_interval_end=0,
	number_of_bins_per_thp=1000):
	'''
	Old binning main.
	'''
	global g_queries_list, g_thp_list, g_dict, g_bin_length_in_time, g_bins_no, g_s, g_f

	g_bin_length_in_time = (millis_interval_start - millis_interval_end) / number_of_bins_per_thp

	g_bins_no = number_of_bins_per_thp

	g_s, g_f = millis_interval_start, millis_interval_end

	if False:
		g_queries_list =\
		tuple(
			map(
				lambda e: (e[0], e[-1],),
				filter(
					lambda e: 'cern' in e[2][0][0],
					pickle.load( open( 'answered_cern_queries.p' , 'rb' ) )
				)
			)
		)
		pickle.dump(
			g_queries_list,
			open('queries_throughput_list.p', 'wb')
		)
		exit(0)

	if True:
		g_queries_list = pickle.load(open('queries_throughput_list.p', 'rb'))

	print('There are ' + str(len(g_queries_list)) + ' queries.')

	if False:
		a_dict = dict()
		for q_time, q_read_size in g_queries_list:
			if q_time in a_dict:
				a_dict[q_time] += q_read_size
			else:
				a_dict[q_time] = q_read_size

		key_list = sorted( a_dict.keys() )

		plt.plot(
			range(len(key_list)),
			tuple(map(lambda k: a_dict[k], key_list))
		)

		plt.savefig('test.png', dpi=150)

		exit(0)

	if True:
		g_thp_list = tuple(
			filter(
				lambda t: t[0] >= first_moment + millis_interval_start,
				pickle.load(open('throughput_dump.p','rb'))
			)
		)

	print('There are ' + str(len(g_thp_list)) + ' throughput values.')

	g_dict = Manager().dict()

	print('Data is loaded ! Now starting process pool !')

	p = Pool(n_proc)

	p.map(get_thp_bin_per_proc, range(len(g_thp_list)))

	p.close()

	pickle.dump(
		dict( g_dict ),
		open('binned_thp_queries_dict.p','wb')
	)

def normalize_five_minute_bins():
	'''
	Normalize bins from dumped file.
	Normalizes PER FEATURE.
	'''
	query_dict = pickle.load(open('binned_thp_queries_dict.p','rb'))

	min_vec = [9576450800000 for _ in range(len(tuple(query_dict.values())[0]))]
	max_vec = [-1 for _ in range(len(tuple(query_dict.values())[0]))]

	min_thp, max_thp = 9576450800000, -1

	for k,value in query_dict.items():
		for i in range(len(value)):
			if min_vec[i] > value[i]: min_vec[i] = value[i]
			if max_vec[i] < value[i]: max_vec[i] = value[i]

		if k[1] < min_thp: min_thp = k[1]
		if k[1] > max_thp: max_thp = k[1]

	key_list = sorted( query_dict.keys() )

	if False:
		a_list = []

		min_a, max_a = 9576450800000, -1

		for key in key_list:
			a_list.append(sum(query_dict[key]))
			if a_list[-1] < min_a: min_a = a_list[-1]
			if a_list[-1] > max_a: max_a = a_list[-1]

		plt.plot(
			range(len(a_list)),
			tuple(
				map(
					lambda e: (e - min_a) / (max_a - min_a),
					a_list
				)
			)
		)

		plt.show()

		exit(0)

	new_array = np.empty((len(key_list), len(query_dict[key_list[0]]) + 1,))

	j = 0

	for key in key_list:
		for i in range(len(query_dict[key])):
			new_array[j,i] = 2 * ( query_dict[key][i] - min_vec[i] ) / ( max_vec[i] - min_vec[i] ) - 1

		new_array[j,-1] = (key[1]-min_thp)/(max_thp-min_thp)

		j+=1

	pickle.dump(
		new_array,
		open('normalized_binned_thp_queries_array.p','wb')
	)

def normalize_five_minute_bins_1():
	'''
	Normalize bins from dumped file.
	Normalizes GLOBALLY.
	'''
	query_dict = pickle.load(open('binned_thp_queries_dict.p','rb'))
	min_bin_rs, max_bin_rs = 9576450800000, -1

	min_thp, max_thp = 9576450800000, -1

	for k,value in query_dict.items():
		for i in range(len(value)):
			if min_bin_rs > value[i]: min_bin_rs = value[i]
			if max_bin_rs < value[i]: max_bin_rs = value[i]

		if k[1] < min_thp: min_thp = k[1]
		if k[1] > max_thp: max_thp = k[1]

	key_list = sorted( query_dict.keys() )

	new_array = np.empty((len(key_list), len(query_dict[key_list[0]]) + 1,))

	j = 0

	for key in key_list:
		for i in range(len(query_dict[key])):
			new_array[j,i] = 2 * ( query_dict[key][i] - min_bin_rs ) / ( max_bin_rs - min_bin_rs ) - 1

		new_array[j,-1] = (key[1]-min_thp)/(max_thp-min_thp)

		j+=1

	pickle.dump(
		new_array,
		open('normalized_binned_thp_queries_array.p','wb')
	)

def analyse_1():
	'''
	Generates data for plotting on reas size, number of queries and
	throughput.
	'''
	read_values_dict = dict()
	q_no_per_time_dict = dict()
	for q_list in filter(
			lambda e: 'cern' in e[2][0][0],
			pickle.load( open( 'answered_cern_queries.p' , 'rb' ) )
		):
		if q_list[0] in read_values_dict:
			read_values_dict[q_list[0]] += q_list[-1]
		else:
			read_values_dict[q_list[0]] = q_list[-1]

		if q_list[0] in q_no_per_time_dict:
			q_no_per_time_dict[q_list[0]] += 1
		else:
			q_no_per_time_dict[q_list[0]] = 1

	read_values_list = sorted(list( read_values_dict.items() ))

	del read_values_dict

	q_no_per_time_list = sorted(list( q_no_per_time_dict.items() ))

	del q_no_per_time_dict

	min_rs = min(map(lambda e: e[1], read_values_list))
	max_rs = max(map(lambda e: e[1], read_values_list))

	min_t = min(map(lambda e: e[0], read_values_list))

	print(min_t)
	exit(0)

	min_rs = min(map(lambda e: e[1], read_values_list))
	max_rs = max(map(lambda e: e[1], read_values_list))
	a_t = (tuple(map(lambda e: (e[0] - min_t) / 60000, read_values_list)),\
		tuple(map(lambda e: (e[1] - min_rs) / ( max_rs - min_rs ), read_values_list)))
	plt.plot(
		a_t[0],
		a_t[1]
	)

	max_rs = max(map(lambda e: e[1], q_no_per_time_list))
	c_t = (tuple(map(lambda e: (e[0] - min_t) / 60000, q_no_per_time_list)),
		tuple(map(lambda e: (e[1] - 1) / ( max_rs - 1 ), q_no_per_time_list)))
	plt.plot(
		c_t[0],
		c_t[1]
	)

	thp_tuple = tuple(
		filter(
			lambda t: t[0] >= read_values_list[0][0],
			pickle.load(open('throughput_dump.p','rb'))
		)
	)
	min_rs = min(map(lambda e: e[1], thp_tuple))
	max_rs = max(map(lambda e: e[1], thp_tuple))
	b_t = (tuple(map(lambda e: (e[0] - min_t) / 60000, thp_tuple)),
		tuple(map(lambda e: (e[1] - min_rs) / ( max_rs - min_rs ), thp_tuple)))
	plt.plot(
		b_t[0],
		b_t[1]
	)

	plt.savefig('test.png', dpi=300)

	pickle.dump(
		a_t + c_t + b_t, open('pipe.p', 'wb')
	)

def analyse_2():
	'''
	Plot data on read size, number of queries and
	throughput.
	'''
	a,b,c,d ,e,f= pickle.load(open('pipe.p','rb'))

	plt.plot(
		a,
		b,
		# 'bo',
		label='read size'
	)

	plt.plot(
		c,
		d,
		# 'ro',
		label='number of queries'
	)

	plt.plot(
		e,
		f,
		# 'go',
		label='throughput'
	)

	for i in range(1440):
		if i % 60 == 0:

			plt.plot([i,i],[0,1])

	plt.legend()

	plt.xlabel('Time in minutes')

	plt.ylabel('Normalized values')

	plt.show()

def split_indexes(
		X,
		output_fn='first_week_train_test_indexes_split.p'
	):
	'''
	Dump train/test split indexes.
	'''

	granularity = 0.1

	number_of_intervals = 10

	intervals_limits_list = [(0,granularity,),]

	while intervals_limits_list[-1][1] < 1:
		intervals_limits_list.append((
			intervals_limits_list[-1][1], intervals_limits_list[-1][1] + granularity
		))

	if number_of_intervals < len(intervals_limits_list):
		intervals_limits_list = intervals_limits_list[:-1]
		intervals_limits_list[-1] = (
			intervals_limits_list[-1][0],
			1
		)

	intervals_limits_list[-1] = (
		intervals_limits_list[-1][0],
		1.1
	)

	print(intervals_limits_list)
	print(len(intervals_limits_list))

	indexes_dict = dict()

	for l_0, l_1 in intervals_limits_list:
		indexes_dict[(l_0, l_1)] = list()
		i = 0
		for line in X:
			if  l_0 <= line[-1] < l_1:
				indexes_dict[(l_0, l_1)].append(i)
			i+=1

	for k,v in indexes_dict.items():
		print(str(k) + ' ' + str(len(v)))

	valid_indexes_list = []

	for indexes_list in indexes_dict.values():
		if len(indexes_list) != 0:
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
		print(len(valid_indexes_list))

	train_indexes_list = list(
		filter(lambda e: e not in valid_indexes_list, range(X.shape[0]))
	)

	print(len(train_indexes_list))

	print(len(valid_indexes_list))

	pickle.dump(
		(train_indexes_list, valid_indexes_list,),
		open(output_fn, 'wb')
	)

def analyse_3():
	'''
	Analyse ratios.
	'''
	a_list = pickle.load(open('answered_cern_queries.p','rb'))

	a =\
	len(
		tuple(
			filter(
				lambda e: 'cern' == e[1] and 'cern' == e[2][0][0],
				a_list
			)
		)
	)
	b =\
	len(
		tuple(
			filter(
				lambda e:'cern' == e[2][0][0],
				a_list
			)
		)
	)
	print('# queries which are emitted by CERN and have CERN as first option: ' + str(a))
	print('# queries which have CERN as first option: ' + str(b))
	print('\tratio: ' + str(a/b))
	print()

	c=\
	len(
		tuple(
			filter(
				lambda e: 'cern' == e[1],
				a_list
			)
		)
	)
	print('# queries which are emitted by CERN and have CERN as first option: ' + str(a))
	print('# queries which are emitted by CERN: ' + str(c))
	print('\tratio: ' + str(a/c))
	print()

def timseseries_analyse_0(thp_dump_list = pickle.load(open('thp_dump_list.p', 'rb'))):
	'''
	Time series analysis.
	'''
	from statsmodels.tsa.seasonal import seasonal_decompose
	import matplotlib.ticker as ticker

	x_list, y_list = list(), list()

	if False:
		prev = thp_dump_list[0][0]

		time_c = (thp_dump_list[1][0] - thp_dump_list[0][0])/1000

		read_size = thp_dump_list[0][1] * (thp_dump_list[1][0] - thp_dump_list[0][0])/1000

		for i in range(2,len(thp_dump_list)):

			time_c += (thp_dump_list[i][0] - thp_dump_list[i-1][0])/1000

			read_size += thp_dump_list[i-1][1] * (thp_dump_list[i][0] - thp_dump_list[i-1][0])/1000

			if thp_dump_list[i][0] - prev >= 1.9 * 60 * 1000:

				x_list.append( prev )
				y_list.append( read_size / time_c )

				x_list.append( thp_dump_list[i][0] )
				y_list.append( read_size / time_c )

				prev = thp_dump_list[i][0]
				time_c, read_size = 0,0
		x_list = tuple( map( lambda e: (e - 1576450800000) / 60000, x_list) )
		min_a,max_a = min(y_list),max(y_list)
		y_list = tuple( map( lambda e: (e - min_a)/(max_a-min_a), y_list) )

	if False:
		y_list =\
		tuple(
			map(
				lambda e: e[1],
				pickle.load(open('thp_dump_list.p', 'rb'))
			)
		)
		x_list =\
		tuple(
			map(
				lambda e: e[0],
				pickle.load(open('thp_dump_list.p', 'rb'))
			)
		)
		x_list =\
		tuple(
			map(
				lambda e: e[0]/1000,
				pickle.load(open('thp_dump_list.p', 'rb'))
			)
		)

	if True:
		y_list =\
		tuple(
			map(
				lambda e: e[1],
				thp_dump_list
			)
		)
		x_list =\
		tuple(
			map(
				lambda e: e[0]/1000,
				thp_dump_list
			)
		)

	import pandas as pd

	dfObj = pd.DataFrame(columns=['Time', 'value',])
	for i in range(len(x_list)):
		dfObj = dfObj.append({'Time': x_list[i], 'value': y_list[i]}, ignore_index=True)

	dfObj.reset_index(inplace=True)

	dfObj['Time'] = pd.to_datetime(dfObj['Time'],unit='s')

	print(dfObj.head())

	result = seasonal_decompose(
		dfObj.value.interpolate(),
		model='additive',
		freq=720
	)

	if False:
		pickle.dump(
			tuple(
				filter(
					lambda e: str(e[1]) != 'nan',
					zip(
						map(lambda e: 1000 * e,x_list),
						result.trend
					)
				)
			),
			open('january_month_throughput_trend.p','wb')
		)
		exit(0)

	if True:
		pickle.dump(
			tuple(
				map(
					lambda e: (e[0], e[1:]),
					filter(
						lambda e: str(e[1]) != 'nan' and str(e[2]) != 'nan' and str(e[3]) != 'nan' ,
						zip(
							map(lambda e: 1000 * e,x_list),
							result.trend,
							result.seasonal,
							result.resid,
						)
					)
				)
			),
			open('january_month_throughput_trend_seasonal_noise.p','wb')
		)
		exit(0)

	if True:
		x_list =\
		tuple(
			map(
				lambda e: e/3600,
				x_list
			)
		)

	# print(result.trend)
	# print(result.seasonal)
	# print(result.resid)
	# print(result.observed)

	# f = open('./to_send/throughput.csv','wt')
	# f.write( 'Time_Stamp_In_Seconds,Original_Throughput,Trend,Seasonality,Noise\n' )
	# for i in range(len(x_list)):
	# 	f.write(
	# 		str(x_list[i]) + ','\
	# 		+ str(y_list[i]) + ','\
	# 		+ str(result.trend[i]) + ','\
	# 		+ str(result.seasonal[i]) + ','\
	# 		+ str(result.resid[i]) + '\n'
	# 	)
	# f.close()

	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	min_a=min(x_list)
	x_list = list( map( lambda e: e - min_a , x_list ) )

	if False:
		ax=plt.subplot(411)
		# plt.plot(range(len(y_list)),result.observed,'bo')
		ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
		plt.plot(x_list,result.observed,label='Throughput')
		time = 0
		while time - 24 < x_list[-1]:
			plt.plot(
				(time,time),
				(min(result.observed),max(result.observed)),
				'r-'
			)
			time+=24
		# plt.xlabel('Time in hours')
		plt.ylabel('MB/s')
		plt.legend()

		ax=plt.subplot(412)
		# plt.plot(range(len(y_list)),result.trend,'bo')
		ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
		plt.plot(x_list,result.trend,label='Trend')
		min_a = min(filter(lambda e: str(e) != 'nan',result.trend))
		max_a = max(filter(lambda e: str(e) != 'nan',result.trend))
		time = 0
		while time - 24 < x_list[-1]:
			plt.plot(
				(time,time),
				(min_a,max_a),
				'r-'
			)
			time+=24
		# plt.xlabel('Time in hours')
		plt.ylabel('MB/s')
		plt.legend()

		ax=plt.subplot(413)
		# plt.plot(range(len(y_list)),result.seasonal,'bo')
		ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
		plt.plot(x_list,result.seasonal,label='Seasonal')
		time = 0
		while time - 24 < x_list[-1]:
			plt.plot(
				(time,time),
				(min(result.seasonal),max(result.seasonal)),
				'r-'
			)
			time+=24
		# plt.xlabel('Time in hours')
		plt.ylabel('MB/s')
		plt.legend()

		ax=plt.subplot(414)
		# plt.plot(range(len(y_list)),result.resid,'bo')
		ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
		plt.plot(x_list,result.resid,label='Noise')
		min_a = min(filter(lambda e: str(e) != 'nan',result.resid))
		max_a = max(filter(lambda e: str(e) != 'nan',result.resid))
		time = 0
		while time - 24 < x_list[-1]:
			plt.plot(
				(time,time),
				(min_a,max_a),
				'r-'
			)
			time+=24
		plt.xlabel('Time in hours')
		plt.ylabel('MB/s')
		plt.legend()

		plt.show()
		plt.clf()

	# 	label='throughput'
	# )

	# plt.legend()

	# plt.xlabel('Index in the Data Set')

	# plt.ylabel('Normalized values')

	# plt.show()

	if True:
		def plot_f(x, y, min_a, max_a, name):
			_, ax = plt.subplots()
			ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
			ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
			ax.plot(
				x,
				y,
				label=name
			)
			time = 0
			while time - 24 < x_list[-1]:
				ax.plot(
					(time,time),
					(min_a,max_a),
					'r-'
				)
				time+=24
			plt.xlabel('Time in hours')
			plt.ylabel('MB/s')
			plt.legend()
			plt.show()
			plt.clf()

		# plot_f(
		# 	x_list,
		# 	result.observed,
		# 	min(result.observed),
		# 	max(result.observed),
		# 	'Throughput'
		# )

		# plot_f(
		# 	x_list,
		# 	result.trend,
		# 	min(filter(lambda e: str(e) != 'nan',result.trend)),
		# 	max(filter(lambda e: str(e) != 'nan',result.trend)),
		# 	'Trend'
		# )

		# plot_f(
		# 	x_list,
		# 	result.seasonal,
		# 	min(result.seasonal),
		# 	max(result.seasonal),
		# 	'Seasonal'
		# )

		plot_f(
			x_list,
			result.resid,
			min(filter(lambda e: str(e) != 'nan',result.resid)),
			max(filter(lambda e: str(e) != 'nan',result.resid)),
			'Noise'
		)

def generate_average_plots():
	if False:
		q_dict = dict()

		for time_stamp, read_size in\
			map(
				lambda e: (e[0],e[-1],),
				filter(
					lambda e: 'cern' in e[2][0][0],
					pickle.load(open('answered_cern_queries.p','rb'))
				)
			):
			if time_stamp in q_dict:
				q_dict[time_stamp] += read_size
			else:
				q_dict[time_stamp] = read_size

		pickle.dump(
			sorted(q_dict.items()),
			open('first_option_cern_only_read_size.p','wb')
		)
	else:
		read_size_list = pickle.load(open('first_option_cern_only_read_size.p','rb'))

		prev = read_size_list[0][0]

		size = read_size_list[0][1]

		x_list = []
		y_list = []

		for time_stamp, read_size in read_size_list[1:]:
			size += read_size

			if time_stamp - prev > 2 * 60 * 1000:
				x_list.append(prev)
				y_list.append( size * 1000 / (time_stamp - prev) )
				x_list.append( time_stamp )
				y_list.append( y_list[-1] )

				prev = time_stamp
				size = 0

		pickle.dump(
			(x_list, y_list),
			open('pipe_1.p','wb')
		)

if __name__ == '__main__':
	first_moment, last_moment = 1576450800000, 1576537200000

	global n_proc
	n_proc = 95

	if False:
		dist, dem = get_matrices(first_moment, last_moment)
		for k0 in dist[0][1].keys():
			for k1 in dist[0][1][k0].keys():
				print(k0 + ' ' + str(k1))
		# # print(dist[0][1][0])
		# # print(dem[0][1][0])

	if False:
		answer_queries_1(first_moment,last_moment)

	if False:
		associate_queries_1(first_moment, last_moment, 'cern')

	if False:
		pickle.dump(
			get_thp_per_proc_1(first_moment, last_moment, 'cern'),
			open('five_minute_throughput_dump.p', 'wb')
		)

	if False:
		assign_interpolated_thp_in_parallel()

	if False:
		normalize_q_t_list()

	if False:
		generate_new_data_set()

	if False:
		verify_query_answering()

	if False:
		get_five_minute_binned_dataset(first_moment)

	if False:
		normalize_five_minute_bins()

	if False:
		analyse_2()

	if False:
		analyse_3()

	if True:
		timseseries_analyse_0(
			tuple(
				filter(
					lambda p: p[0] >= 1578960000000,
					map(
						lambda p: (int(p[0])*1000, float(p[1]),),
						tuple(
							csv.reader(
								open(
									'january_month_throughput.csv','rt'
								)
							)
						)[1:]
					)
				)
			)
		)

	if False:
		normalize_five_minute_bins_1()

	if False:
		generate_average_plots()