import pickle
import csv
from multiprocessing import Pool, Manager, Process, Lock
from multiprocessing.sharedctypes import RawArray
import os
import shutil
import json
from functools import reduce
import gc
from collections import namedtuple
from copy import deepcopy
import random
from scipy import interpolate
import itertools

DATA_SETS_FOLDER = '/optane/mipopa/data_set_dumps/'

def get_unanswered_queries_main(client_name):
	'''
	Parses a raw query file.
	'''
	pickle.dump(
		tuple(
			map(
				lambda line: (\
					int(line[0]),
					line[3].lower(),
					tuple(\
						map(
							lambda e: tuple(e.split('::')[1:]),
							line[4].split(';')
						)
					),
					int(line[2]),
				),
				filter(
					lambda line: client_name in line[-1],
					map(
						lambda line: line[:-1] + [line[-1].lower(),],
						csv.reader(open('apicommands.log'),delimiter=',')
					)
				)
			)
		),
		open('unanswered_cern_queries.p', 'wb')
	)

def get_unanswered_queries_main_1(client_name):
	'''
	Parses a raw query file.
	'''
	global file_variable

	if False:
		file_variable = '#epoch time,file name,size,accessed from,sorted replica list\n'\
			+ '1579215600000,/alice/cern.ch/user/h/hosokawa/myWorkingDirUPC_MB2_pass1/myOutputDirUPC_MB2_pass1/000246053/022/AnalysisResults.root,6996489,UPB,ALICE::CERN::EOS;ALICE::GSI::SE2\n'\
			+ '1579215600000,/alice/cern.ch/user/h/hosokawa/myWorkingDirUPC_MB2_pass1/myOutputDirUPC_MB2_pass1/000246053/147/AnalysisResults.root,1556821,GSI,ALICE::GSI::SE2;ALICE::NIHAM::EOS\n'\
			+ '1579215600000,/alice/cern.ch/user/h/hosokawa/myWorkingDirUPC_MB2_pass1/myOutputDirUPC_MB2_pass1/000246053/153/AnalysisResults.root,1340549,GSI,ALICE::GSI::SE2;ALICE::CERN::EOS\n'\
			+ '1579215600000,/alice/cern.ch/user/h/hosokawa/myWorkingDirUPC_MB2_pass1/myOutputDirUPC_MB2_pass1/000246053/154/AnalysisResults.root,1507233,GSI,ALICE::CERN::SE2;ALICE::NIHAM::EOS\n'
	if True:
		file_variable = open('apicommands.log','rt').read()
		# len(file_variable) = 43433223135
	if False:
		file_variable = open('test.log','rt').read()

	file_variable = file_variable.split('\n')[1:]
	if len(file_variable[-1]) < 5: file_variable = file_variable[:-1]

	print(len(file_variable))
	for i in\
		map(
			lambda line: line[:-1] + [line[-1].lower(),],
			map(
				lambda line: line.split(','),
				file_variable[:10]
			)
		): print(i)

	print('Will start working !')

	if True:
		print(
			len(
				tuple(
					filter(
						lambda line: client_name in line[-1],
						map(
							lambda line: line[:-1] + [line[-1].lower(),],
							map(
								lambda line: line.split(','),
								file_variable
							)
						)
					)
				)
			)
		)
		exit(0)

	a = tuple(
		map(
			lambda line: (\
				int(line[0]),
				line[3].lower(),
				tuple(\
					map(
						lambda e: tuple(e.split('::')[1:]),
						line[4].split(';')
					)
				),
				int(line[2]),
			),
			filter(
				lambda line: client_name in line[-1],
				map(
					lambda line: line[:-1] + [line[-1].lower(),],
					map(lambda line: line.split(','),file_variable)
				)
			)
		)
	)

	print(len(a))

	pickle.dump(
		a,
		open('unanswered_cern_queries.p', 'wb')
	)

def process_per_proc(i):
	'''
	Parses a raw query file per CPU core. Used inside a processes pool.
	'''
	# f = open( './unanswered_query_dump_folder/' + str(i) + '.json' , 'wt' )
	f = open( g_dump_folder_name + str(i) + '.json' , 'wt' )
	json.dump(
		tuple(
			map(
				lambda line: (\
					int(line[0]),
					line[3].lower(),
					tuple(\
						map(
							lambda e: tuple(e.split('::')[1:]),
							line[4].split(';')
						)
					),
					int(line[2]),
				),
				filter(
					lambda line: g_client_name in line[-1],
					map(
						lambda line: line[:-1] + [line[-1].lower(),],
						map(
							lambda line: line.split(','),
							file_variable[\
								g_idexes_pairs_tuple[i][0] : g_idexes_pairs_tuple[i][1]\
							]
						)
					)
				)
			)
		),
		f,
	)
	print('Finished extracting for ' + str(i))
	f.close()
	gc.collect()

def analyse_per_proc(i):
	'''
	Counts queries per CPU core. Used inside a processes pool.
	'''
	l =\
	len(
		tuple(
			filter(
				lambda line: g_client_name in line[-1],
				map(
					lambda line: line[:-1] + [line[-1].lower(),],
					map(
						lambda line: line.split(','),
						file_variable[\
							g_idexes_pairs_tuple[i][0] : g_idexes_pairs_tuple[i][1]\
						]
					)
				)
			)
		)
	)
	print('Between ' + str(g_idexes_pairs_tuple[i][0]) + ' and ' + str(g_idexes_pairs_tuple[i][1]) + ' there are ' + str(l))
	return l

def get_unanswered_queries_main_2(client_name, dump_folder_name, input_file_name='apicommands.log'):
	'''
	Parses a raw query file. Uses a pool of processes.
	'''

	# 40GB -> 450m49.952s

	global file_variable, g_idexes_pairs_tuple, g_client_name, g_dump_folder_name

	g_dump_folder_name = dump_folder_name

	g_client_name = client_name

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

	if True: Pool(n_proc).map(process_per_proc, range(len(g_idexes_pairs_tuple)))
	if False:
		print('Total number of q: '\
			+ str(
				reduce(
					lambda acc, x: acc + x,
					Pool(n_proc).map(analyse_per_proc, range(len(g_idexes_pairs_tuple)))
				)
			)
		)

def get_matrices_time_moments_lists(first_moment, last_moment, matrices_folder):
	'''
	Trims and returns matrices coording to time moments.
	'''
	filename_tuple = tuple(os.listdir(matrices_folder))
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

def get_matrices(first_moment, last_moment, matrices_folder):

	distance_list, demotion_list = get_matrices_time_moments_lists(
		first_moment,
		last_moment,
		matrices_folder
	)

	dist_res_list = []
	for time_moment in distance_list:
		dist_res_list.append((time_moment,get_dist_dict_by_time(time_moment,matrices_folder),))

	dem_res_list = []
	for time_moment in demotion_list:
		dem_res_list.append((time_moment,get_dem_dict_by_time(time_moment,matrices_folder),))

	return dist_res_list, dem_res_list

def binary_search_matrixes(matrix_list, time_moment, left_i, right_i):
	if right_i - left_i == 1:
		if matrix_list[left_i][0] <= time_moment < matrix_list[right_i][0]:
			return left_i
		return right_i
	mid_i = (left_i + right_i) // 2
	if time_moment < matrix_list[mid_i][0]:
		return binary_search_matrixes(
			matrix_list,
			time_moment,
			left_i,
			mid_i
		)
	return binary_search_matrixes(
		matrix_list,
		time_moment,
		mid_i,
		right_i
	)

def answer_per_proc(aaaa):
	'''
	Answers queries based on the 2 matrices per CPU core. Used inside a processes pool.
	'''
	queries_list = list(
		map(
			lambda q_line:\
				(
					q_line[0],
					q_line[1],
					tuple(map(lambda e: tuple(e),q_line[2])),
					q_line[3],
				),
			filter(
				lambda p: p[0] >= g_first_moment,
				json.load(
					open(g_input_folder + g_file_list[aaaa],'rt')
				)
			)
		)
	)

	dist_i = 0
	dem_i = 0

	res_list = list()

	for i in range(len(queries_list)):

		found_client = False

		ref_dict = g_dist_list[-1][1]
		for k in ref_dict.keys():
			if queries_list[i][1] in k or k in queries_list[i][1]:
				client_name = k
				found_client = True
				break

		if not found_client:
			continue

		if queries_list[i][0] >= g_dist_list[-1][0]:
			local_dist_dict = ref_dict[client_name]
		else:
			while not ( g_dist_list[dist_i][0] <= queries_list[i][0] < g_dist_list[dist_i+1][0] ):
				dist_i+=1
			local_dist_dict = g_dist_list[dist_i][1][client_name]
			# local_dist_dict = g_dist_list[\
			# 	binary_search_matrixes(g_dist_list,queries_list[i][0],0,len(g_dist_list)-1)\
			# ][1][client_name]

		if queries_list[i][0] >= g_dem_list[-1][0]:
			local_dem_dict = g_dem_list[-1][1]
		else:
			while not ( g_dem_list[dem_i][0] <= queries_list[i][0] < g_dem_list[dem_i+1][0] ):
				dem_i+=1
			local_dem_dict = g_dem_list[dem_i][1]
			# local_dem_dict = g_dem_list[\
			# 	binary_search_matrixes(g_dem_list,queries_list[i][0],0,len(g_dem_list)-1)\
			# ][1]

		se_list = []
		j = 0
		for se in queries_list[i][2]:
			se_list.append(
				(
					local_dist_dict[se] + local_dem_dict[se],
					j
				)
			)
			j+=1
		se_list.sort(key=lambda p: p[0])

		if False:
			queries_list[i] = (
				queries_list[i][0],
				queries_list[i][1],
				tuple(map(lambda p: queries_list[i][2][p[1]], se_list)),
				queries_list[i][3],
			)
		if True:
			res_list.append(
				(
					queries_list[i][0],
					queries_list[i][1],
					tuple(map(lambda p: queries_list[i][2][p[1]], se_list)),
					queries_list[i][3],
				)
			)

	json.dump(
		tuple(res_list),
		open(g_output_folder + g_file_list[aaaa], 'wt'),
	)

	del queries_list

	print('Finished for: ' + g_file_list[aaaa])

	gc.collect()

def analyse_before_answer_0(aaaa):
	'''
	Runs inside processes pool.
	'''
	return\
	len(
		tuple(
			filter(
				lambda p: p[0] >= g_first_moment and len(p[1]) <= 2,
				json.load(
					open(g_input_folder + g_file_list[aaaa],'rt')
				)
			)
		)
	)

def get_answered_queries_main_1(first_moment, last_moment, input_folder, output_folder, matrices_folder):
	'''
	Answers in parallel.
	'''

	'''
	16GB -> 6m35.779s
			5m4.435s
	'''

	global g_dist_list, g_dem_list, g_file_list, g_first_moment, g_input_folder, g_output_folder

	g_input_folder = input_folder
	g_output_folder = output_folder

	g_first_moment = first_moment

	if True:
		g_dist_list, g_dem_list = get_matrices(first_moment, last_moment, matrices_folder)

		print('There are ' + str(len(g_dist_list)) + ' distance matrices !')
		print('There are ' + str(len(g_dem_list)) + ' demotion matrices !')

	g_file_list = tuple(\
		map(
			lambda fn: fn,
			os.listdir(input_folder)
		)
	)

	print('Will start pool !')

	if True:
		Pool(n_proc).map(answer_per_proc, range(len(g_file_list)))
	if False:
		print(sum(Pool(n_proc).map(analyse_before_answer_0, range(len(g_file_list)))))

def analyse_dist_dem(first_moment, last_moment):
	matrixes_list = os.listdir('remote_host/log_folder')

	dist_list = tuple(
		map(
			lambda fn: int(fn.split('_')[0]),
			filter(
				lambda fn: '_distance' in fn,
				matrixes_list
			)
		)
	)

	dem_list = tuple(
		map(
			lambda fn: int(fn.split('_')[0]),
			filter(
				lambda fn: '_demotion' in fn,
				matrixes_list
			)
		)
	)

	print(str(first_moment) + ' ' + str(last_moment))

	print(str(min(dist_list)) + ' ' + str(max(dist_list)))

	print(str(min(dem_list)) + ' ' + str(max(dem_list)))

def get_first_option_cern_0(input_folder, output_file):
	'''
	16GB -> 128m53.432s
	'''
	json.dump(
		sorted(
			reduce(
				lambda acc,x: acc + list(x),
				map(
					lambda fn:\
						filter(
							lambda q_line: 'cern' in q_line[2][0][0],
							json.load(
								open(
									input_folder + fn,
									'rt'
								)
							)
						),
					sorted(os.listdir(input_folder))
				),
				list()
			)
		),
		open(
			output_file,
			'wt'
		)
	)

def get_first_option_cern_1(input_folder, output_file):
	a_list = list()

	i = 0

	for filtered_generator in map(
			lambda fn:\
				filter(
					lambda q_line: 'cern' in q_line[2][0][0],
					json.load(
						open(
							input_folder + fn,
							'rt'
						)
					)
				),
			sorted(os.listdir(input_folder))
		):

		print(i)

		a_list.extend( filtered_generator )

		i+=1

	json.dump(
		sorted(a_list),
		open(
			output_file,
			'wt'
		)
	)

def get_first_option_cern_2(input_folder, output_file):
	json.dump(
		sorted(
			itertools.chain.from_iterable(
				map(
					lambda fn:\
						filter(
							lambda q_line: 'cern' in q_line[2][0][0],
							json.load(
								open(
									input_folder + fn,
									'rt'
								)
							)
						),
					sorted(os.listdir(input_folder))
				)
			)
		),
		open(
			output_file,
			'wt'
		)
	)

def get_five_minute_binned_dataset_1(
	first_moment,
	millis_interval_start=4000000,
	millis_interval_end=0,
	number_of_bins_per_thp=1000):

	# queries_list = json.load(open('first_option_cern.json', 'rt'))

	# print('There are ' + str(len(queries_list)) + ' queries.')

	thp_list = sorted(\
		tuple(
			filter(
				lambda t: t[0] >= first_moment + millis_interval_start,
				pickle.load(open('thp_dump_list.p','rb'))
			)
		)
	)

	print('There are ' + str(len(thp_list)) + ' throughput values.')

	Bin_Element = namedtuple('Bin_El',['ind','fi_t','la_t','thp_t','bin_list','thp_v'])

	initial_bin_list = [0 for _ in range(number_of_bins_per_thp)]

	bin_length_in_time = (millis_interval_start - millis_interval_end) / number_of_bins_per_thp

	result_list = []

	queue_list = []

	thp_i = 0

	q_index = 0

	for time_stamp, _, _, read_size in json.load(open('first_option_cern.json', 'rt')):

		if q_index % 100000 == 0:
			print('Reached query index: ' + str(q_index))

			if len(queue_list) >= 10:
				a_str = ''
				for i in random.sample(range(len(queue_list)),10):
					a_str += str(queue_list[i].bin_list[queue_list[i].ind]) + ' '
				print('\t' + a_str)

		while thp_i < len(thp_list) and thp_list[thp_i][0] - millis_interval_start <= time_stamp < thp_list[thp_i][0]:

			bin_i = 0
			bin_t = thp_list[thp_i][0] - millis_interval_start
			while True:
				if bin_i >= g_number_of_bins_per_thp or bin_t <= time_stamp < bin_t + bin_length_in_time:

					queue_list.append(
						Bin_Element(
							bin_i,
							bin_t,
							bin_t + bin_length_in_time,
							thp_list[thp_i][0],
							deepcopy(initial_bin_list),
							thp_list[thp_i][1],
						)
					)

					break

				bin_t += bin_length_in_time

				bin_i += 1

			thp_i += 1

		q_i = 0

		while q_i < len(queue_list):

			if time_stamp < queue_list[q_i].thp_t:

				if q_i - 1 >= 0: queue_list = queue_list[q_i:]

				break

			result_list.append( queue_list[q_i] )

			q_i += 1

		for q_i in range(len(queue_list)):
			if not ( queue_list[q_i].fi_t <= time_stamp < queue_list[q_i].la_t ):
				bin_i = queue_list[q_i].ind
				bin_t = queue_list[q_i].fi_t
				while bin_i < number_of_bins_per_thp:

					if queue_list[q_i].fi_t <= time_stamp < queue_list[q_i].la_t:

						queue_list[q_i] = Bin_Element(
							ind=bin_i,
							fi_t=bin_t,
							la_t=bin_t + bin_length_in_time,
							bin_list=queue_list[q_i].bin_list,
							thp_v=queue_list[q_i].thp_v
						)

						break

					bin_i += 1
					bin_t += bin_length_in_time

		for bin_el in queue_list:
			bin_el.bin_list[bin_el.ind] += read_size

		q_index += 1

	pickle.dump(
		tuple(
			map(
				lambda bin_el: bin_el.bin_list + [bin_el.thp_v,],
				result_list
			)
		),
		open('ten_gigs_bins.p','wb')
	)

def reduce_query_list_size():
	json.dump(
		tuple(
			map(
				lambda p: (p[0],p[-1],),
				json.load(open('first_option_cern.json', 'rt'))
			)
		),
		open('first_opt_cern_only_read_value.json','wt')
	)

def reduce_query_list_size_1(input_fn, output_fn):
	'''
	27 minutes
	22 min
	'''
	d = dict()
	for tm, _, _, rs in json.load(open(input_fn,'rt')):
		if tm not in d:
			d[tm] = rs
		else:
			d[tm] += rs
	json.dump(
		sorted(d.items()),
		open(output_fn,'wt')
	)

def reduce_query_list_size_2(input_folder, output_fn):
	'''
	10m43.830s
	'''
	d = dict()
	for generator_from_file in\
		map(
			lambda fn:\
				filter(
					lambda q_line: 'cern' in q_line[2][0][0],
					json.load(
						open(
							input_folder + fn,
							'rt'
						)
					)
				),
			sorted(os.listdir(input_folder))
		):
		for tm, _, _, rs in generator_from_file:
			if tm not in d:
				d[tm] = rs
			else:
				d[tm] += rs
	json.dump(
		sorted(d.items()),
		open(output_fn,'wt')
	)

def bins_per_proc(ii):
	Bin_Element = namedtuple('Bin_El',['ind','fi_t','la_t','thp_t','bin_list','thp_v', 'thp_i'])

	slice_start = g_slice_list[ ii ][ 0 ]
	slice_end = g_slice_list[ ii ][ 1 ]

	initial_bin_list = [0 for _ in range(g_number_of_bins_per_thp)]

	queue_list = []

	thp_i = len(g_thp_list) - 1
	while thp_i >= 0 and g_query_list[ slice_start ][ 0 ] < g_thp_list[ thp_i ][ 0 ]:
		thp_i -= 1
	thp_i+=1

	thp_la_i = thp_i
	while thp_la_i < len(g_thp_list) and g_thp_list[ thp_la_i ][ 0 ] - g_millis_interval_start <= g_query_list[ slice_end - 1 ][ 0 ]:
		thp_la_i+=1

	# print(str(thp_i) + ' ' + str(thp_la_i))

	q_index = 0
	for time_stamp, read_size in g_query_list[ slice_start : slice_end ]:
	# for time_stamp, read_size in g_query_list[g_slice_list[ii][0]:g_slice_list[ii][0]+10]:
		if q_index % 100000 == 0:
			print(str(os.getpid()) + ': Reached query index: ' + str(q_index) + '/' + str(g_slice_list[ii][1]-g_slice_list[ii][0]))

		# Write read size to shared memory.
		q_i = 0

		while q_i < len(queue_list):

			if time_stamp < queue_list[ q_i ].thp_t:

				queue_list = queue_list[ q_i : ]

				break

			g_lock_list[ queue_list[ q_i ].thp_i ].acquire()

			for jj in range(g_number_of_bins_per_thp):

				g_result_list[ queue_list[ q_i ].thp_i ][ jj ] =\
					g_result_list[ queue_list[ q_i ].thp_i ][ jj ]\
					+ queue_list[ q_i ].bin_list[ jj ]

			g_lock_list[ queue_list[ q_i ].thp_i ].release()

			q_i += 1

		# Add new thp bin elements.
		while thp_i < thp_la_i and g_thp_list[thp_i][0] - g_millis_interval_start <= time_stamp < g_thp_list[thp_i][0]:

			bin_i = 0

			bin_t = g_thp_list[thp_i][0] - g_millis_interval_start

			while True:
				if bin_t <= time_stamp < bin_t + g_bin_length_in_time:

					queue_list.append(
						Bin_Element(
							bin_i,
							bin_t,
							bin_t + g_bin_length_in_time,
							g_thp_list[thp_i][0],
							deepcopy(initial_bin_list),
							g_thp_list[thp_i][1],
							thp_i,
						)
					)

					break

				bin_t += g_bin_length_in_time

				bin_i += 1

			thp_i += 1

		# Increase bin index if necessary.
		for q_i in range(len(queue_list)):
			if not ( queue_list[q_i].fi_t <= time_stamp < queue_list[q_i].la_t ):

				bin_i = queue_list[q_i].ind

				bin_t = queue_list[q_i].fi_t

				while bin_i < g_number_of_bins_per_thp:

					if bin_t <= time_stamp < bin_t + g_bin_length_in_time:

						queue_list[q_i] = Bin_Element(
							bin_i,
							bin_t,
							bin_t + g_bin_length_in_time,
							queue_list[q_i].thp_t,
							queue_list[q_i].bin_list,
							queue_list[q_i].thp_v,
							queue_list[q_i].thp_i
						)

						break

					bin_i += 1

					bin_t += g_bin_length_in_time

		for bin_el in queue_list:
			bin_el.bin_list[bin_el.ind] += read_size

		q_index += 1

	for bin_el in queue_list:
		g_lock_list[ bin_el.thp_i ].acquire()

		for jj in range(g_number_of_bins_per_thp):

			g_result_list[ bin_el.thp_i ][ jj ] =\
				g_result_list[ bin_el.thp_i ][ jj ]\
				+ bin_el.bin_list[ jj ]

		g_lock_list[ bin_el.thp_i ].release()

def get_five_minute_binned_dataset_2(
	first_moment,
	last_moment,
	millis_interval_start=4000000,
	millis_interval_end=0,
	number_of_bins_per_thp=1000,):

	global g_result_list, g_thp_list, g_query_list, g_slice_list,\
		g_number_of_bins_per_thp, g_bin_length_in_time, g_millis_interval_start,\
		g_lock_list

	DEBUG_FLAG = False
	if not DEBUG_FLAG:
		if True:
			g_query_list = sorted(json.load(open('first_opt_cern_only_read_value.json', 'rt')))
		if False:
			g_query_list = sorted(pickle.load(open('queries_throughput_list.p', 'rb')))
	else:
		g_query_list,_ = pickle.load(open('debug_small.p','rb'))

	if not DEBUG_FLAG:
		if False:
			g_thp_list = sorted(pickle.load(open('thp_dump_list.p','rb')))
		if False:
			g_thp_list = sorted(pickle.load(open('throughput_dump.p','rb')))
		if False:
			g_thp_list =\
			tuple(
				map(
					lambda p: (1000*int(p[0]), float(p[1]),),
					tuple(
						csv.reader(
							open('january_month_throughput.csv','rt')
						)
					)[1:]
				)
			)
		if False:
			g_thp_list=pickle.load(
				open(
					'january_month_throughput_trend.p',
					'rb'
				)
			)
		if False:
			g_thp_list = sorted(pickle.load(open('thp_list_100k_one_week.p','rb')))[50000:]
		if True:
			g_thp_list=pickle.load(
				open(
					'january_month_throughput_trend_seasonal_noise.p',
					'rb'
				)
			)
	else:
		_,g_thp_list = pickle.load(open('debug_small.p','rb'))

	g_query_list = sorted(g_query_list)
	g_thp_list = sorted(g_thp_list)

	# Adjust query list to make it past the first moment.
	i = 0
	while g_query_list[i][0] < first_moment:
		i += 1
	g_query_list = g_query_list[i:]

	# Increase thp time to be bigger that the first q time to make sure
	# only queries that affect the throughput are taken into consideration.
	if g_thp_list[0][0] - millis_interval_start < g_query_list[0][0]:
		i = 0
		while g_thp_list[i][0] - millis_interval_start < g_query_list[0][0]:
			i += 1
		g_thp_list = g_thp_list[i:]

	# Eliminate redundant first queries.
	i = 0
	while g_query_list[i][0] < g_thp_list[0][0] - millis_interval_start:
		i += 1
	g_query_list = g_query_list[i:]

	# Eliminate redundant last thp.
	if g_query_list[-1][0] <= g_thp_list[-1][0]:
		i = len(g_thp_list) - 1
		while g_query_list[-1][0] <= g_thp_list[i][0]:
			i -= 1
		g_thp_list = g_thp_list[:i+1]

	# Eliminate redundant last queries.
	i = len(g_query_list) - 1
	while g_query_list[i][0] >= g_thp_list[-1][0]:
		i-=1
	g_query_list = g_query_list[:i+1]

	if False:
		pickle.dump(
			(
				tuple(map(lambda e: e[0], g_query_list)),
				tuple(map(lambda e: e[0], g_thp_list)),
			),
			open('pipe_1.p','wb')
		)
		exit(0)

	g_result_list = [RawArray('d',number_of_bins_per_thp*[0,]) for _ in range(len(g_thp_list))]

	g_lock_list = [Lock() for _ in range(len(g_thp_list))]

	if False:
		g_thp_list = g_thp_list[:100]

		i = len(g_query_list) - 1
		while g_query_list[i][0] > g_thp_list[-1][0]:
			i-=1

		g_query_list = g_query_list[:i+1]

		pickle.dump(
			(
				g_query_list,
				g_thp_list,
			),
			open(
				'debug_small.p',
				'wb'
			)
		)
		exit(0)

	print('There are ' + str(len(g_thp_list)) + ' throughput values.')
	q_count = len(g_query_list)
	print('There are ' + str(q_count) + ' queries values.')

	a_list = [q_count//n_proc for _ in range(n_proc)]
	for i in range(q_count%n_proc):
		a_list[i] += 1
	g_slice_list = [(0, a_list[0],),]
	for i in range(1, n_proc):
		g_slice_list.append((
			g_slice_list[-1][1],
			g_slice_list[-1][1] + a_list[i],
		))

	del a_list

	g_number_of_bins_per_thp = number_of_bins_per_thp

	g_bin_length_in_time = (millis_interval_start - millis_interval_end) / number_of_bins_per_thp

	g_millis_interval_start = millis_interval_start

	p = Pool(n_proc)

	print('Will start pool !')

	p.map(
		bins_per_proc,
		range(n_proc)
	)

	p.close()

	p.join()

	del g_query_list
	del g_slice_list

	if False:
		out_fn = 'first_week_data_set.json'
	if False:
		out_fn = 'first_week_data_set_trend.json'
	if False:
		out_fn = 'first_week_data_set_site_thp.json'
	if False:
		out_fn = 'first_week_data_set_100k_interp_part_1.json'
	if True:
		out_fn = 'first_week_data_set_trend_seasonal_noise.json'

	json.dump(
		tuple(
			map(
				lambda ind: tuple(g_result_list[ind]) + (g_thp_list[ind][1],),
				range(len(g_thp_list))
			)
		),
		open(out_fn,'wt')
	)

def normalize_data_set(data_set_tuple, out_fn='first_week_normalized_data_set.json'):
	min_rs = data_set_tuple[0][0]
	max_rs = data_set_tuple[0][0]

	min_thp,max_thp = data_set_tuple[0][-1],data_set_tuple[0][-1]

	for e in data_set_tuple:
		a = min(e[:-1])
		if a < min_rs: min_rs = a
		a = max(e[:-1])
		if a > max_rs: max_rs = a
		if min_thp < e[-1]: min_thp = e[-1]
		if max_thp > e[-1]: max_thp = e[-1]

	json.dump(
		tuple(
			map(
				lambda e:\
					tuple(
						map(
							lambda a: 2*(a-min_rs)/(max_rs-min_rs)-1,\
							e[:-1]
						)
					)\
					+(
						1 - ( e[-1] - min_thp ) / ( max_thp - min_thp ),\
					),
				data_set_tuple
			)
		),
		open(out_fn,'wt')
	)

def normalize_filter_split_from_cont_list(bins_list, remaining_bins_no, time_window, output_fn):
	import itertools
	import numpy as np

	min_rs, max_rs = bins_list[0][0][0], bins_list[0][0][0]

	min_thp, max_thp = bins_list[0][0][-1], bins_list[0][0][-1]

	for b_list in bins_list:
		for e in b_list:
			a = min(e[:-1])
			if a < min_rs: min_rs = a
			a = max(e[:-1])
			if a > max_rs: max_rs = a
			if min_thp > e[-1]: min_thp = e[-1]
			if max_thp < e[-1]: max_thp = e[-1]

	if False:
		bins_list =\
		list(
			map(
				lambda b_list:\
					list(
						map(
							lambda e:\
								tuple(
									map(
										lambda a: 2*(a-min_rs)/(max_rs-min_rs)-1,
										e[:-1]
									)
								)\
								+ ( 2*(e[-1]-min_thp)/(max_thp-min_thp)-1 , ),
							b_list
						)
					),
				bins_list
			)
		)

		top_k_set =\
		set(
			map(
				lambda p: p[0],
				sorted(
						enumerate(
							map(
								lambda ind:
									float(
										np.correlate(
											list(map(lambda e: e[ind], itertools.chain.from_iterable(bins_list))),
											list(map(lambda e: e[-1], itertools.chain.from_iterable(bins_list))),
										)
									),
								range(len(bins_list[0][0])-1)
							)
						)
					,
					key=lambda p: p[1],
					reverse=True
				)[:remaining_bins_no]
			)
		)

		bins_list = list(
			map(
				lambda b_list:\
					list(
						map(
							lambda bins:\
								tuple(
									map(
										lambda p: p[1],
										filter(
											lambda e: e[0] in top_k_set,
											enumerate(bins[:-1])
										)
									)
								)\
								+ ( ( bins[-1] + 1 ) / 2 , ),
							b_list
						)
					),
				bins_list
			)
		)

	if True:
		bins_list =\
		list(
			map(
				lambda b_list:\
					list(
						map(
							lambda e:\
								tuple(
									map(
										lambda a: 2*(a-min_rs)/(max_rs-min_rs)-1,
										e[:-1]
									)
								)\
								+ ( (e[-1]-min_thp)/(max_thp-min_thp) , ),
							b_list
						)
					),
				bins_list
			)
		)

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

			mean_v = 0

			for l in range( j , j + time_window , 1 ):

				new_bin_list[-1].append(list())

				for e in b_list[ l ]:
					new_bin_list[-1][-1].append(e)

				mean_v += b_list[ l ][-1]

			mean_v /= time_window

			if True:
				mean_v = b_list[ j + time_window - 1 ][-1]

			for k in indexes_dict.keys():

				if k[0] <= mean_v < k[1]:

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

	if True:
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

	json.dump(
		(train_x_list, train_y_list, valid_x_list, valid_y_list,),
		open(
			output_fn, 'wt'
		)
	)

def normalize_data_set_1(in_filename, out_fn='first_week_normalized_data_set.json'):
	data_set_tuple = json.load(open(in_filename,'rt'))

	min_rs = data_set_tuple[0][0]
	max_rs = data_set_tuple[0][0]

	min_thp,max_thp = data_set_tuple[0][-1][0],data_set_tuple[0][-1][0]

	for e in data_set_tuple:
		a = min(e[:-1])
		if a < min_rs: min_rs = a
		a = max(e[:-1])
		if a > max_rs: max_rs = a
		if min_thp < e[-1][0]: min_thp = e[-1][0]
		if min_thp < e[-1][1]: min_thp = e[-1][1]
		if min_thp < e[-1][2]: min_thp = e[-1][2]
		if max_thp > e[-1][0]: max_thp = e[-1][0]
		if max_thp > e[-1][1]: max_thp = e[-1][1]
		if max_thp > e[-1][2]: max_thp = e[-1][2]

	json.dump(
		tuple(
			map(
				lambda e:\
					tuple(
						map(
							lambda a: 2*(a-min_rs)/(max_rs-min_rs)-1,\
							e[:-1]
						)
					)\
					+(
						(
							1 - ( e[-1][0] - min_thp ) / ( max_thp - min_thp ),
							1 - ( e[-1][1] - min_thp ) / ( max_thp - min_thp ),
							1 - ( e[-1][2] - min_thp ) / ( max_thp - min_thp ),
						),
					),
				data_set_tuple
			)
		),
		open(out_fn,'wt')
	)

def split_indexes(X,output_fn='first_week_train_test_indexes_split.p'):
	'''
	Dump train/test split indexes.
	'''
	if False:
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

	if True:
		intervals_limits_list = [
			( 0 , 0.05),
			( 0.05 , 0.1 ),
			( 0.1 , 0.15 ),
			( 0.15 , 0.2 ),
			( 0.2 , 0.25 ),
			( 0.25 , 0.3 ),
			( 0.3 , 0.35 ),
			( 0.35 , 0.4 ),
			( 0.4 , 0.45 ),
			( 0.45 , 0.5 ),
			( 0.5 , 0.55 ),
			( 0.55 , 0.6 ),
			( 0.6 , 0.65 ),
			( 0.65 , 0.7 ),
			( 0.7 , 0.75 ),
			( 0.75 , 0.8 ),
			( 0.8 , 0.85 ),
			( 0.85 , 0.9 ),
			( 0.9 , 0.95 ),
			( 0.95 , 1.05 ),
		]

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
		filter(lambda e: e not in valid_indexes_list, range(len(X)))
	)

	print(len(train_indexes_list))

	print(len(valid_indexes_list))

	pickle.dump(
		(train_indexes_list, valid_indexes_list,),
		open(output_fn, 'wb')
	)
	print()

def analyse_site_throughput():
	from csv import reader
	a = tuple(map(lambda p: (int(p[0]), float(p[1])), tuple(reader(open('january_month_throughput.csv','rt')))[1:]))

	a = tuple(
		filter(
			lambda e: 1579264801.390 <= e[0] < 1579875041.000,
			a
		)
	)

	print(len(a))
	print(a[0])
	print(a[-1])

	import matplotlib.pyplot as plt

	plt.plot(
		tuple( map( lambda e: e[0] / (2 * 60) , a ) ),
		tuple( map( lambda e: e[1] , a ) )
	)

	plt.show()

def get_interpolated_throughput_from_csv(first_moment, last_moment):
	a,b=\
	reduce(
		lambda acc, x: ( acc[0] + [x[0],] , acc[1] + [x[1],] , ),
		filter(
			lambda p: first_moment - 10 <= p[0] <= last_moment + 10,
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
		),
		(list(),list())
	)

	thp_func = interpolate.interp1d(a,b)

	c = list()
	time_interv = (last_moment - first_moment)/100000
	t = min(a)
	while t <= max(a):
		c.append(
			(
				t,
				float(thp_func(t)),
			)
		)
		t+=time_interv

	pickle.dump(
		c,
		open(
			'thp_list_100k_one_week.p',
			'wb'
		)
	)

def get_comparison_thp_trend_plots():
	import matplotlib.pyplot as plt
	trend_list = json.load(open('first_week_normalized_data_set_trend.json','rt'))
	ax=plt.subplot(2,1,1)
	plt.plot(
		range(len(trend_list)),
		tuple(map(lambda e: e[-2], trend_list)),
		label='Normalized Last Bin'
	)
	plt.plot(
		range(len(trend_list)),
		tuple(map(lambda e: 2 * e[-1] - 1, trend_list)),
		label='Normalized Trend'
	)
	plt.ylabel('Normalized value')
	ax.legend()
	del trend_list

	original_list = json.load(open('first_week_normalized_data_set_site_thp.json','rt'))
	ax=plt.subplot(2,1,2)
	plt.plot(
		range(len(original_list)),
		tuple(map(lambda e: e[-2], original_list)),
		label='Normalized Last Bin'
	)
	plt.plot(
		range(len(original_list)),
		tuple(map(lambda e: 2 * e[-1] - 1, original_list)),
		label='Normalized Throughput'
	)
	plt.ylabel('Normalized value')
	plt.xlabel('Index in the Data Set')
	ax.legend()
	del original_list

	plt.show()

def get_total_number_of_ses():
	a = reduce(
		lambda acc,x: acc | x,
		map(
			lambda fn:\
				reduce(
					lambda acc, x: acc | set( map(lambda e: tuple(e), x[2]) ),
					filter(
						lambda q_line: 'cern' in map(lambda e: e[0],q_line[2]),
						json.load(
							open(
								'./answered_query_dump_folder/' + fn,
								'rt'
							)
						)
					),
					set()
				),
			os.listdir('./answered_query_dump_folder/')
		),
		set()
	)

	print(a)
	print(len(a))

def get_all_throughputs(first_moment, last_moment,):

	spool_dir_path = 'remote_host/spool/'

	return_dict = dict()

	client_name_sets_dict = dict()

	i = 0

	for processed_generator in\
		map(
			lambda filename:\
				map(
					lambda e: (int(e[1]), e[2].lower(), e[5].split('_OUT')[0], float(e[6]),),
					filter(
						lambda e: len(e) == 7\
							and '_OUT_freq' not in e[5]\
							and '_IN' not in e[5],
						map(
							lambda line: line.split('\t'),
							open(spool_dir_path+filename,'rt').read().split('\n')
						)
					)
				),
			filter(
				lambda e:\
					'.done' in e\
					and first_moment - 3600000 <= int(e[:-5]) < last_moment + 3600000,
				os.listdir(spool_dir_path)
			)
		):
		if i % 100 == 0:
			print(i)
		for tm, se_name, from_name, value in processed_generator:
			if se_name not in return_dict:
				return_dict[se_name] = { tm : { from_name : value } }
			else:
				if tm not in return_dict[se_name]:
					return_dict[se_name][tm] = { from_name : value }
				else:
					return_dict[se_name][tm][from_name] = value
			if se_name not in client_name_sets_dict:
				client_name_sets_dict[se_name] = set( ( from_name , ) )
			else:
				client_name_sets_dict[se_name].add( from_name )
		i+=1

	for se_name in return_dict.keys():
		for tm in return_dict[se_name]:
			for from_name in client_name_sets_dict[se_name]:
				if from_name not in return_dict[se_name][tm]:
					return_dict[se_name][tm][from_name] = 0

	pickle.dump(
		return_dict,
		open(
			'first_week_Se_name_Tm_From_name_dict.p',
			'wb'
		)
	)

def extract_client_from_big_thp_dict(se_name, dump_name):
	for se_name_a, tm_dict in pickle.load(open('first_week_Se_name_Tm_From_name_dict.p','rb')).items():
		if se_name in se_name_a:
			pickle.dump(
				tm_dict,
				open(
					dump_name,
					'wb'
				)
			)
			break

def bins_per_proc_3(ii):
	Bin_Element = namedtuple('Bin_El',['ind','fi_t','la_t','thp_t','bin_list','thp_v', 'thp_i'])

	slice_start = g_slice_list[ ii ][ 0 ]
	slice_end = g_slice_list[ ii ][ 1 ]

	initial_bin_list = [0 for _ in range(g_number_of_bins_per_thp)]

	queue_list_dict = dict()
	for k in g_result_list[0].keys():
		queue_list_dict[k] = list()

	thp_i = len(g_thp_list) - 1
	while thp_i >= 0 and g_query_list[ slice_start ][ 0 ] < g_thp_list[ thp_i ][ 0 ]:
		thp_i -= 1
	thp_i+=1

	thp_la_i = thp_i
	while thp_la_i < len(g_thp_list) and g_thp_list[ thp_la_i ][ 0 ] - g_millis_interval_start <= g_query_list[ slice_end - 1 ][ 0 ]:
		thp_la_i+=1

	q_index = 0
	for time_stamp, emittent_cl_name, read_size in g_query_list[ slice_start : slice_end ]:
		if q_index % 100000 == 0:
			print(str(os.getpid()) + ': Reached query index: ' + str(q_index) + '/' + str(g_slice_list[ii][1]-g_slice_list[ii][0]))

		# Write read size to shared memory.
		for cl_name, queue_list in queue_list_dict.items():
			q_i = 0

			while q_i < len(queue_list):

				# Eliminate from queue if necessary.
				if time_stamp < queue_list[ q_i ].thp_t:

					queue_list = queue_list[ q_i : ]

					break

				if g_lock_iterable[ queue_list[ q_i ].thp_i ] is not None:
					g_lock_iterable[ queue_list[ q_i ].thp_i ].acquire()

				for jj in range(g_number_of_bins_per_thp):

					g_result_list[ queue_list[ q_i ].thp_i ][cl_name][ jj ] =\
						g_result_list[ queue_list[ q_i ].thp_i ][cl_name][ jj ]\
						+ queue_list[ q_i ].bin_list[ jj ]

				if g_lock_iterable[ queue_list[ q_i ].thp_i ] is not None:
					g_lock_iterable[ queue_list[ q_i ].thp_i ].release()

				q_i += 1

		# Add new thp bin elements.
		while thp_i < thp_la_i and g_thp_list[thp_i][0] - g_millis_interval_start <= time_stamp < g_thp_list[thp_i][0]:

			bin_i = 0

			bin_t = g_thp_list[thp_i][0] - g_millis_interval_start

			while True:
				if bin_t <= time_stamp < bin_t + g_bin_length_in_time:

					for queue_list in queue_list_dict.values():
						queue_list.append(
							Bin_Element(
								bin_i,
								bin_t,
								bin_t + g_bin_length_in_time,
								g_thp_list[thp_i][0],
								deepcopy(initial_bin_list),
								g_thp_list[thp_i][1],
								thp_i,
							)
						)

					break

				bin_t += g_bin_length_in_time

				bin_i += 1

			thp_i += 1

		# Increase bin index if necessary.
		for queue_list in queue_list_dict.values():
			for q_i in range(len(queue_list)):
				if not ( queue_list[q_i].fi_t <= time_stamp < queue_list[q_i].la_t ):

					bin_i = queue_list[q_i].ind

					bin_t = queue_list[q_i].fi_t

					while bin_i < g_number_of_bins_per_thp:

						if bin_t <= time_stamp < bin_t + g_bin_length_in_time:

							queue_list[q_i] = Bin_Element(
								bin_i,
								bin_t,
								bin_t + g_bin_length_in_time,
								queue_list[q_i].thp_t,
								queue_list[q_i].bin_list,
								queue_list[q_i].thp_v,
								queue_list[q_i].thp_i
							)

							break

						bin_i += 1

						bin_t += g_bin_length_in_time

		for bin_el in queue_list_dict[emittent_cl_name]:
			bin_el.bin_list[bin_el.ind] += read_size

		q_index += 1

	for cl_name, queue_list in queue_list_dict.items():
		for bin_el in queue_list:

			if g_lock_iterable[ bin_el.thp_i ] is not None:
				g_lock_iterable[ bin_el.thp_i ].acquire()

			for jj in range(g_number_of_bins_per_thp):

				g_result_list[ bin_el.thp_i ][cl_name][ jj ] =\
					g_result_list[ bin_el.thp_i ][cl_name][ jj ]\
					+ bin_el.bin_list[ jj ]

			if g_lock_iterable[ bin_el.thp_i ] is not None:
				g_lock_iterable[ bin_el.thp_i ].release()

def get_five_minute_binned_dataset_3(
	first_moment,
	last_moment,
	millis_interval_start=4000000,
	millis_interval_end=0,
	number_of_bins_per_thp=100,):

	global g_result_list, g_thp_list, g_query_list, g_slice_list,\
		g_number_of_bins_per_thp, g_bin_length_in_time, g_millis_interval_start,\
		g_lock_iterable

	if False:

		client_names_set = set()
		g_query_list = list()
		for a, b, _, d in json.load(open('first_option_cern.json','rt')):
			g_query_list.append((a,b,d,))
			client_names_set.add(b)
		pickle.dump(
			g_query_list,
			open('first_option_cern_tm_emittent_rs.p','wb')
		)
		pickle.dump(
			client_names_set,
			open('first_option_cern_emittents_set.p','wb')
		)
		exit(0)

	if True:
		client_names_set = pickle.load(open('first_option_cern_emittents_set.p','rb'))
		g_query_list = pickle.load(open('first_option_cern_tm_emittent_rs.p','rb'))

	if True:
		g_thp_list = pickle.load(open('first_week_cern_thp_per_client.p','rb')).items()

	print('Finished loading data !')

	g_query_list = sorted(g_query_list)
	g_thp_list = sorted(g_thp_list)

	# Adjust query list to make it past the first moment.
	i = 0
	while g_query_list[i][0] < first_moment:
		i += 1
	g_query_list = g_query_list[i:]

	# Increase thp time to be bigger that the first q time to make sure
	# only queries that affect the throughput are taken into consideration.
	if g_thp_list[0][0] - millis_interval_start < g_query_list[0][0]:
		i = 0
		while g_thp_list[i][0] - millis_interval_start < g_query_list[0][0]:
			i += 1
		g_thp_list = g_thp_list[i:]

	# Eliminate redundant first queries.
	i = 0
	while g_query_list[i][0] < g_thp_list[0][0] - millis_interval_start:
		i += 1
	g_query_list = g_query_list[i:]

	# Eliminate redundant last thp.
	if g_query_list[-1][0] <= g_thp_list[-1][0]:
		i = len(g_thp_list) - 1
		while g_query_list[-1][0] <= g_thp_list[i][0]:
			i -= 1
		g_thp_list = g_thp_list[:i+1]

	# Eliminate redundant last queries.
	i = len(g_query_list) - 1
	while g_query_list[i][0] >= g_thp_list[-1][0]:
		i-=1
	g_query_list = g_query_list[:i+1]


	print('There are ' + str(len(g_thp_list)) + ' throughput values.')
	q_count = len(g_query_list)
	print('There are ' + str(q_count) + ' queries values.')

	a_list = [q_count//n_proc for _ in range(n_proc)]
	for i in range(q_count%n_proc):
		a_list[i] += 1
	g_slice_list = [(0, a_list[0],),]
	for i in range(1, n_proc):
		g_slice_list.append((
			g_slice_list[-1][1],
			g_slice_list[-1][1] + a_list[i],
		))

	del a_list

	g_result_list = list()
	g_lock_iterable = list()
	for tm, _ in g_thp_list:

		is_race_condition_flag = False
		for _, s_end in g_slice_list[:-1]:
			if tm - millis_interval_start <= g_query_list[s_end][0] < tm:
				is_race_condition_flag = True
				break

		if is_race_condition_flag:
			g_lock_iterable.append(Lock())
		else:
			g_lock_iterable.append(None)

		new_d = dict()
		for name in client_names_set:
			new_d[name] = RawArray('d',number_of_bins_per_thp*[0,])
		g_result_list.append(new_d)

	g_number_of_bins_per_thp = number_of_bins_per_thp

	g_bin_length_in_time = (millis_interval_start - millis_interval_end) / number_of_bins_per_thp

	g_millis_interval_start = millis_interval_start

	p = Pool(n_proc)

	print('Will start pool !')

	p.map(
		bins_per_proc_3,
		range(n_proc)
	)

	p.close()

	p.join()

	del g_query_list
	del g_slice_list
	del g_lock_iterable

	if True:
		out_fn = 'first_week_data_set_cern_all_clients.json'

	se_name_iterable = sorted( g_thp_list[0][1].keys() )

	res_list = list()
	i = 0
	for small_dict in g_result_list:
		x_t = list()
		for cl_name in client_names_set:
			x_t += list(small_dict[cl_name])

		y_t = list()
		for se_name in se_name_iterable:
			y_t.append(g_thp_list[i][1][se_name])

		res_list.append(
			( x_t , y_t )
		)
		i+=1

	json.dump(
		res_list,
		open(out_fn,'wt')
	)

def bins_per_proc_4(ii):
	'''
	Extracts bins per CPU core. It is runs inside a process pool. It uses a
	sliding window to go through the temporally ordered queries and assign
	their read size to bins which in turn are assigned to logged throughputs.
	'''
	Bin_Element = namedtuple('Bin_El',['ind','fi_t','la_t','thp_t','ipc_elem','thp_v', 'thp_i'])

	slice_start = g_slice_list[ ii ][ 0 ]
	slice_end = g_slice_list[ ii ][ 1 ]

	thp_i = len(g_thp_list) - 1
	while thp_i >= 0 and g_query_list[ slice_start ][ 0 ] < g_thp_list[ thp_i ][ 0 ]:
		thp_i -= 1
	thp_i+=1

	thp_la_i = thp_i
	while thp_la_i < len(g_thp_list) and g_thp_list[ thp_la_i ][ 0 ] - g_millis_interval_start <= g_query_list[ slice_end - 1 ][ 0 ]:
		thp_la_i+=1

	queue_list = list()

	q_index = 0
	for time_stamp, emittent_cl_name, read_size in g_query_list[ slice_start : slice_end ]:
		if q_index % 100000 == 0:
			print(str(os.getpid()) + ': Reached query index: ' + str(q_index) + '/' + str(g_slice_list[ii][1]-g_slice_list[ii][0]))

		# Eliminate from queue if necessary.
		q_i = 0
		while q_i < len(queue_list):
			if time_stamp < queue_list[ q_i ].thp_t:

				queue_list = queue_list[ q_i : ]

				break

			q_i += 1

		# Add new thp bin elements.
		while thp_i < thp_la_i and g_thp_list[thp_i][0] - g_millis_interval_start <= time_stamp < g_thp_list[thp_i][0]:

			bin_i = 0

			bin_t = g_thp_list[thp_i][0] - g_millis_interval_start

			while True:
				if bin_t <= time_stamp < bin_t + g_bin_length_in_time:

					queue_list.append(
						Bin_Element(
							bin_i,
							bin_t,
							bin_t + g_bin_length_in_time,
							g_thp_list[thp_i][0],
							g_ipc_list[thp_i],
							g_thp_list[thp_i][1],
							thp_i,
						)
					)

					break

				bin_t += g_bin_length_in_time

				bin_i += 1

			thp_i += 1

		# Increase bin index if necessary.
		for q_i in range(len(queue_list)):
			if not ( queue_list[q_i].fi_t <= time_stamp < queue_list[q_i].la_t ):

				bin_i = queue_list[q_i].ind

				bin_t = queue_list[q_i].fi_t

				while bin_i < g_number_of_bins_per_thp:

					if bin_t <= time_stamp < bin_t + g_bin_length_in_time:

						queue_list[q_i] = Bin_Element(
							bin_i,
							bin_t,
							bin_t + g_bin_length_in_time,
							queue_list[q_i].thp_t,
							queue_list[q_i].ipc_elem,
							queue_list[q_i].thp_v,
							queue_list[q_i].thp_i
						)

						break

					bin_i += 1

					bin_t += g_bin_length_in_time

		for bin_el in queue_list:

			if bin_el.ipc_elem.lock is not None:
				bin_el.ipc_elem.lock.acquire()

			bin_el.ipc_elem.arrays_dict[emittent_cl_name][bin_el.ind] += read_size

			if bin_el.ipc_elem.lock is not None:
				bin_el.ipc_elem.lock.release()

		q_index += 1

def get_five_minute_binned_dataset_4(
	first_moment,
	last_moment,
	millis_interval_start=4000000,
	millis_interval_end=0,
	number_of_bins_per_thp=100,):
	'''
	Assigns queries into bins and bins to throughput values.

	first_moment,last_moment
		Make up the temporal points between which bins are put.

	millis_interval_start,millis_interval_end
		Define the interval in miliseconds the bis span. For a throughput value received
		at time t, the interval is [ t - millis_interval_start ; t - millis_interval_end ).

	number_of_bins_per_thp
		Number of equally sized bins to split the interval into.
	'''

	global g_thp_list, g_query_list, g_slice_list,\
		g_number_of_bins_per_thp, g_bin_length_in_time, g_millis_interval_start,\
		g_ipc_list

	if False:

		client_names_set = set()
		g_query_list = list()
		for a, b, _, d in json.load(open('first_option_cern.json','rt')):
			g_query_list.append((a,b,d,))
			client_names_set.add(b)
		pickle.dump(
			g_query_list,
			open('first_option_cern_tm_emittent_rs.p','wb')
		)
		pickle.dump(
			client_names_set,
			open('first_option_cern_emittents_set.p','wb')
		)
		exit(0)

	if True:
		client_names_set = pickle.load(open('first_option_cern_emittents_set.p','rb'))
		g_query_list = pickle.load(open('first_option_cern_tm_emittent_rs.p','rb'))

	if True:
		g_thp_list = pickle.load(open('first_week_cern_thp_per_client.p','rb')).items()

	print('Finished loading data !')

	g_query_list = sorted(g_query_list)
	g_thp_list = sorted(g_thp_list)

	# Adjust query list to make it past the first moment.
	i = 0
	while g_query_list[i][0] < first_moment:
		i += 1
	g_query_list = g_query_list[i:]

	# Increase thp time to be bigger that the first q time to make sure
	# only queries that affect the throughput are taken into consideration.
	if g_thp_list[0][0] - millis_interval_start < g_query_list[0][0]:
		i = 0
		while g_thp_list[i][0] - millis_interval_start < g_query_list[0][0]:
			i += 1
		g_thp_list = g_thp_list[i:]

	# Eliminate redundant first queries.
	i = 0
	while g_query_list[i][0] < g_thp_list[0][0] - millis_interval_start:
		i += 1
	g_query_list = g_query_list[i:]

	# Eliminate redundant last thp.
	if g_query_list[-1][0] <= g_thp_list[-1][0]:
		i = len(g_thp_list) - 1
		while g_query_list[-1][0] <= g_thp_list[i][0]:
			i -= 1
		g_thp_list = g_thp_list[:i+1]

	# Eliminate redundant last queries.
	i = len(g_query_list) - 1
	while g_query_list[i][0] >= g_thp_list[-1][0]:
		i-=1
	g_query_list = g_query_list[:i+1]


	print('There are ' + str(len(g_thp_list)) + ' throughput values.')
	q_count = len(g_query_list)
	print('There are ' + str(q_count) + ' queries values.')

	a_list = [q_count//n_proc for _ in range(n_proc)]
	for i in range(q_count%n_proc):
		a_list[i] += 1
	g_slice_list = [(0, a_list[0],),]
	for i in range(1, n_proc):
		g_slice_list.append((
			g_slice_list[-1][1],
			g_slice_list[-1][1] + a_list[i],
		))

	del a_list

	IPC_Element = namedtuple('IPC_El',['arrays_dict', 'lock'])

	g_ipc_list = list()
	for tm, _ in g_thp_list:

		is_race_condition_flag = False
		for _, s_end in g_slice_list[:-1]:
			if tm - millis_interval_start <= g_query_list[s_end][0] < tm:
				is_race_condition_flag = True
				break

		new_d = dict()
		for name in client_names_set:
			new_d[name] = RawArray('d',number_of_bins_per_thp*[0,])

		g_ipc_list.append(
			IPC_Element(
				arrays_dict=new_d,
				lock=Lock() if is_race_condition_flag else None,
			)
		)

	g_number_of_bins_per_thp = number_of_bins_per_thp

	g_bin_length_in_time = (millis_interval_start - millis_interval_end) / number_of_bins_per_thp

	g_millis_interval_start = millis_interval_start

	p = Pool(n_proc)

	print('Will start pool !')

	p.map(
		bins_per_proc_4,
		range(n_proc)
	)

	p.close()

	p.join()

	del g_query_list
	del g_slice_list

	if True:
		out_fn = 'first_week_data_set_cern_all_clients.json'

	thp_cl_name_iterable = sorted( g_thp_list[0][1].keys() )

	res_list = list()
	i = 0
	for ipc_el in g_ipc_list:
		x_t = list()
		for cl_name in client_names_set:
			x_t += list(ipc_el.arrays_dict[cl_name])

		y_t = list()
		for se_name in thp_cl_name_iterable:
			y_t.append(g_thp_list[i][1][se_name])

		res_list.append(
			( x_t , y_t )
		)

		i+=1

	json.dump(
		res_list,
		open(out_fn,'wt')
	)

def normalize_data_set_2():
	if True:
		data_set_list = json.load(
			open(
				'first_week_data_set_cern_all_clients.json',
				'rt'
			)
		)
	min_rs,max_rs =\
		min( map( lambda p: min(p[0]) , data_set_list ) ),\
		max( map( lambda p: max(p[0]) , data_set_list ) )

	min_thp, max_thp =\
		min( map( lambda p: min(p[1]) , data_set_list ) ),\
		max( map( lambda p: max(p[1]) , data_set_list ) )

	json.dump(
		tuple(
			map(
				lambda p:\
					(
						tuple(map(lambda e: (e - min_rs) / (max_rs - min_rs) , p[0])),
						tuple(map(lambda e: (e - min_thp) / (max_thp - min_thp) , p[1])),
					),
				data_set_list
			)
		),
		open('first_week_normalized_data_set_cern_all_clients.json','wt')
	)

def compare_unanswered_answered_counts():
	print(
		'Unanswered count: ' +\
		str(
			sum(
				map(
					lambda fn:\
						len(
							json.load(
								open(
									'/data/mipopa/unanswered_query_dump_folder/'+fn,
									'rt'
								)
							)
						),
					os.listdir( '/data/mipopa/unanswered_query_dump_folder/' )
				)
			)
		)
	)
	print(
		'Answered count: ' +\
		str(
			sum(
				map(
					lambda fn:\
						len(
							json.load(
								open(
									'/data/mipopa/answered_query_dump_folder/'+fn,
									'rt'
								)
							)
						),
					os.listdir( '/data/mipopa/answered_query_dump_folder/' )
				)
			)
		)
	)

def time_moments_from_unanswered_queries(folder_name):
	min_tm, max_tm =\
	reduce(
		lambda pair_acc, x:\
			( min((pair_acc[0], x)) , max((pair_acc[1], x)) , ) if len(pair_acc) != 0\
			else (x,x,),
		json.load(open(folder_name + '200.json','rt')) +  json.load(open(folder_name + '0.json','rt')),
		tuple()
	)

	print(min_tm)
	print(max_tm)

def analyse_unasnwered_per_proc(i):
	cl_count, rs_size, number_of_qs, total_rs = 0, 0, 0, 0
	for _ , cl , _ , rs in json.load(open('unanswered_query_dump_folder_1/' + filename_list[i],'rt')):
		if len(cl) <= 2:
			cl_count += 1
			rs_size += rs
		number_of_qs += 1
		total_rs += rs

	return (cl_count, rs_size, number_of_qs,total_rs,)

def analyse_unasnwered_querries():

	global filename_list

	filename_list = os.listdir('unanswered_query_dump_folder_1/')

	res_list = Pool(n_proc).map(
		analyse_unasnwered_per_proc,
		range(len(filename_list))
	)

	print(
		'# 0-client: ' + str(sum(map(lambda p: p[0], res_list))) + '/' + str(sum(map(lambda p: p[2], res_list)))\
		+ ' -> ' + str( sum(map(lambda p: p[0], res_list)) / sum(map(lambda p: p[2], res_list)) )
	)

	print(
		'rs: ' + str(sum(map(lambda p: p[1], res_list))) + '/' + str(sum(map(lambda p: p[3], res_list)))\
		+ ' -> ' + str( sum(map(lambda p: p[1], res_list)) / sum(map(lambda p: p[3], res_list)) )
	)

def external_bin_main(query_file, output_file, thp_file, time_len, bins_no):
	from binning_main import get_five_minute_binned_dataset_5

	query_list = tuple(
		json.load(
			open(
				query_file,
				'rt'
			)
		)
	)

	first_moment, last_moment = query_list[0][0], query_list[-1][0]

	print('first_moment = ' + str(first_moment) + ' , last_moment = ' + str(last_moment))

	if True:
		thp_gen = csv.reader(open(thp_file))

		next(thp_gen)

		a_list, b_list = list(), list()

		for line in thp_gen:
			if len(line) == 2:
				a = 1000 * int(line[0])
				if first_moment + time_len <= a < last_moment:
					a_list.append(a)
					b_list.append(float(line[1]))

		thp_func = interpolate.interp1d(a_list,b_list)

		thp_list = list()

		t = a_list[0]

		while t <= a_list[-1]:

			thp_list.append(
				(
					t,
					float(thp_func(t)),
				)
			)

			t += 120000

		thp_list = tuple(thp_list)

		del thp_func
		del a_list
		del b_list
		del t

	if False:
		from statsmodels.tsa.seasonal import seasonal_decompose

		thp_gen = csv.reader(open(thp_file))

		next(thp_gen)

		a_list, b_list = list(), list()

		for line in thp_gen:
			if len(line) == 2:
				a_list.append(1000 * int(line[0]))
				b_list.append(float(line[1]))

		thp_func = interpolate.interp1d(a_list,b_list)

		thp_list = list()

		t = a_list[0]

		while t <= a_list[-1]:

			thp_list.append(
				(
					t,
					float(thp_func(t)),
				)
			)

			t += 120000

		del a_list
		del b_list
		del t

		thp_list=\
		list(
			map(
				lambda p: ( p[1][0] , p[0] , ) ,
				filter(
					lambda p: str(p[0]) != 'nan'\
						and first_moment + time_len <= p[1][0] < last_moment,
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

	gc.collect()

	get_five_minute_binned_dataset_5(
		query_list,
		thp_list,
		output_file,
		first_moment,
		last_moment,
		n_proc,
		time_len,
		0,
		bins_no,
	).join()

def test_binning_algo(query_file, thp_file, time_len, bins_no):
	query_list =\
	json.load(
		open(
			query_file,
			'rt'
		)
	)[:1000]


	first_moment, last_moment = query_list[0][0], query_list[-1][0]

	print('first_moment = ' + str(first_moment) + ' , last_moment = ' + str(last_moment))

	thp_gen = csv.reader(open(thp_file))

	next(thp_gen)

	a_list, b_list = list(), list()

	for line in thp_gen:
		if len(line) == 2:
			a = 1000 * int(line[0])
			if first_moment + time_len <= a < last_moment:
				a_list.append(a)
				b_list.append(float(line[1]))

	thp_func = interpolate.interp1d(a_list,b_list)

	thp_list = list()

	t = a_list[0]

	while t <= a_list[-1]:

		thp_list.append(
			(
				t,
				float(thp_func(t)),
			)
		)

		t += 120000

	thp_list = tuple(thp_list)

	del thp_func
	del a_list
	del b_list
	del t

	bins_list = []

	for _ in range(len(thp_list)):
		bins_list.append( bins_no * [0,] )

	bin_time_interval = time_len / bins_no

	for q_time, rs in query_list:
		i = 0
		for t_time, thp in thp_list:
			if t_time - time_len <= q_time < t_time:

				t = t_time - time_len

				for j in range(bins_no):

					if t <= q_time < t + bin_time_interval:

						bins_list[i] += rs

					t += bin_time_interval

			i+=1

if __name__ == '__main__':
	if True:
		# first one week
		first_moment, last_moment = 1579264801390, 1579875041000
	if False:
		# one day
		first_moment, last_moment = 1576450800000, 1576537200000
	if False:
		# second week
		first_moment, last_moment = 1580511600000, 1581289199000
	if False:
		# third week
		first_moment, last_moment = 1581289199000, 1581697742233

	global n_proc

	n_proc = 95

	if False: get_unanswered_queries_main_1('cern')

	if False:
		get_unanswered_queries_main_2(
			'cern',
			'./unanswered_query_dump_folder_2/',
			'/data/mipopa/apicommands3.log',
		)

	if False:
		get_answered_queries_main_1(
			first_moment,
			last_moment,
			'./unanswered_query_dump_folder_1/',
			'./answered_query_dump_folder_1/',
			# '/data/mipopa/remote_host/log_folder/',
			'./remote_host_1/log_folder/',
		)

	if False: analyse_dist_dem(first_moment, last_moment)

	if False:
		get_first_option_cern_2(
			'/data/mipopa/answered_query_dump_folder_0/',
			'./week_0_first_option_cern_queries.json'
		)

	if False: get_five_minute_binned_dataset_2(first_moment,last_moment)

	if False:
		a=tuple(map(lambda p: tuple(p[1]) + (p[0][1],),sorted(pickle.load(open('binned_thp_queries_dict.p','rb')).items())))
		b=pickle.load(open('binned_thp_queries_dict_for_comparison.p','rb'))
		print(a==b)
		for i in range(len(a)):
				if a[i] != b[i]:
					break
		print(i)

	if False: reduce_query_list_size()

	if False:
		normalize_data_set_1(
			'first_week_data_set_trend_seasonal_noise.json',
			'first_week_normalized_data_set_trend_seasonal_noise.json'
		)

	if False:
		import json
		a=json.load(open('data_set.json','rt'))

	if False:
		import matplotlib.pyplot as plt
		a,b = pickle.load(open('pipe_1.p','rb'))

		plt.plot(
			tuple(map(lambda e: e / (60000), a)),
			[0 for _ in range(len(a))]
		)

		plt.plot(
			tuple(map(lambda e: e / (60000), b)),
			[1 for _ in range(len(b))]
		)

		plt.show()

	if False:
		split_indexes(
			json.load(open('three_weeks_normalized_data_set.json','rt')),
			'three_weeks_train_test_indexes_split.p'
		)

	if False:
		analyse_site_throughput()

	if False:
		get_interpolated_throughput_from_csv(first_moment,last_moment)

	if False:
		get_comparison_thp_trend_plots()

	if False:
		get_total_number_of_ses()

	if False:
		get_all_throughputs(first_moment,last_moment)

	if False:
		extract_client_from_big_thp_dict(
			'cern',
			'first_week_cern_thp_per_client.p'
		)

	if False:
		get_five_minute_binned_dataset_4(first_moment,last_moment)

	if False:
		normalize_data_set_2()

	if False:
		a_list = tuple(
			map(
				lambda p: sum(p[1]),
				json.load(open('first_week_normalized_data_set_cern_all_clients.json','rt'))
			)
		)
		min_a, max_a = min(a_list), max(a_list)
		a_list = tuple(
			map(
				lambda e: ((e-min_a) / (max_a-min_a),),
				a_list
			)
		)
		split_indexes(a_list, 'first_week_train_test_indexes_split_cern_all_clients.p')

	if False:
		time_moments_from_unanswered_queries('/optane/mipopa/unanswered_query_dump_folder_1/')

	if False:
		analyse_new_raw_querries()

	if False:
		import sys

		reduce_query_list_size_1(
			'week_'+sys.argv[1]+'_first_option_cern_queries.json',
			'week_'+sys.argv[1]+'_first_option_cern_only_read.json'
		)

	if False:
		'''
		0m25.955s
		'''
		normalize_data_set(
			json.load(open('week_0_first_option_cern_data_set.json','rt'))\
			+ json.load(open('week_1_first_option_cern_data_set.json','rt'))\
			+ json.load(open('week_2_first_option_cern_data_set.json','rt')),
			'three_weeks_normalized_data_set.json'
		)

	if True:
		out_0_fn, out_1_fn, out_2_fn =\
			'week_0_first_option_cern_data_set.json',\
			'week_1_first_option_cern_data_set.json',\
			'week_2_first_option_cern_data_set.json',

	if False:
		external_bin_main(
			DATA_SETS_FOLDER+'week_0_first_option_cern_only_read.json',
			DATA_SETS_FOLDER+out_0_fn,
			'january_month_throughput.csv',
			4000000,
			90,

		)
		external_bin_main(
			DATA_SETS_FOLDER+'week_1_first_option_cern_only_read.json',
			DATA_SETS_FOLDER+out_1_fn,
			'february_month.csv',
			4000000,
			90,
		)
		external_bin_main(
			DATA_SETS_FOLDER+'week_2_first_option_cern_only_read.json',
			DATA_SETS_FOLDER+out_2_fn,
			'february_month.csv',
			4000000,
			90,
		)
		if False:
			normalize_data_set(
				json.load(open(DATA_SETS_FOLDER+'week_0_first_option_cern_4k_bins_data_set.json','rt'))\
				+ json.load(open(DATA_SETS_FOLDER+'week_1_first_option_cern_4k_bins_data_set.json','rt'))\
				+ json.load(open(DATA_SETS_FOLDER+'week_2_first_option_cern_4k_bins_data_set.json','rt')),
				DATA_SETS_FOLDER+'three_weeks_normalized_4k_bins_data_set.json'
			)
			split_indexes(
				json.load(open('three_weeks_normalized_trend_data_set.json','rt')),
				'three_weeks_train_test_indexes_split_trend.p'
			)

	if False:
		external_bin_main(
			'week_0_first_option_cern_only_read.json',
			'week_0_first_option_cern_data_set_.json',
			'january_month_throughput.csv',
			4000000,
		)
		normalize_data_set(
			json.load(open('week_0_first_option_cern_data_set_FOR_COMPARISON.json','rt')),
			'week_0_first_option_cern_normalized_data_set_FOR_COMPARISON.json'
		)

	if True:
		normalize_filter_split_from_cont_list(
			[
				json.load(open(DATA_SETS_FOLDER+out_0_fn,'rt')),
				json.load(open(DATA_SETS_FOLDER+out_1_fn,'rt')),
				json.load(open(DATA_SETS_FOLDER+out_2_fn,'rt'))
			],
			2000,
			35,
			DATA_SETS_FOLDER+'three_weeks_old_split_thp_data_set.json',
		)

	if False:
		analyse_unasnwered_querries()


