from multiprocessing import Pool
import os
import itertools
import json
import pickle
import matplotlib.pyplot as plt
import csv
from collections import namedtuple
import matplotlib.pyplot as plt

# Pickle files containing unanswered queries containg CERN
# as a valid option.
UNANSWERED_PATH_TUPLE = (\
	'/data/mipopa/unanswered_query_dump_folder_0/',\
	'/optane/mipopa/unanswered_query_dump_folder_1/',\
	'/optane/mipopa/unanswered_query_dump_folder_2/',\
)

# Pickle files containing answered queries
ANSWERED_PATH_TUPLE = (\
	'/data/mipopa/unanswered_query_dump_folder_0/',\
	'/optane/mipopa/unanswered_query_dump_folder_1/',\
	'/optane/mipopa/unanswered_query_dump_folder_2/',\
)

ASSIGNED_THROUGHPUT_CSV_FILES = (\
	'january_month_throughput.csv',\
	'february_month.csv',\
	'february_month.csv',\
)

WEEK_TIME_MOMENTS = (\
	(1579215600000, 1579875041000),\
	(1580511600000, 1581289199000),\
	(1581289200000, 1581980399000),\
)

def define_global_tags():
	global q_count_tag, min_read_tag, max_read_tag, total_read_tag, avg_read_tag,\
		min_options_count_tag, max_options_count_tag, tm_stats_dict_tag
	q_count_tag = 0
	min_read_tag = 1
	max_read_tag = 2
	total_read_tag = 3
	avg_read_tag = 4
	min_options_count_tag = 5
	max_options_count_tag = 6
	tm_stats_dict_tag = 7

def get_stats_per_proc(i):
	'''
	Extracts some statistics from the a file.
	'''
	tm_dict = dict()

	min_options, max_options = None, None

	for q_list in json.load(open(filename_list[i],'rt')):

		if min_options is None or len( q_list[2] ) < min_options:
			min_options = len( q_list[2] )

		if max_options is None or len( q_list[2] ) > max_options:
			max_options = len( q_list[2] )

		if q_list[0] in tm_dict:
			tm_dict[ q_list[0] ][ q_count_tag ] += 1
			if q_list[-1] < tm_dict[ q_list[0] ][ min_read_tag ]:
				tm_dict[ q_list[0] ][ min_read_tag ] = q_list[-1]
			if q_list[-1] > tm_dict[ q_list[0] ][ max_read_tag ]:
				tm_dict[ q_list[0] ][ max_read_tag ] = q_list[-1]
			tm_dict[ q_list[0] ][ total_read_tag ] += q_list[-1]
		else:
			tm_dict[ q_list[0] ] = {
				q_count_tag : 1, 				# Count of queries per time moment
				min_read_tag : q_list[-1],		# Minimum read size per time moment
				max_read_tag : q_list[-1],		# Maximum read size per time moment
				total_read_tag : q_list[-1],	# Total read size per time moment
			}


	return tm_dict , min_options, max_options

def analyse_shape_of_input_data_main():
	'''
	Gathers some statistics in parallel from query files.
	'''
	global filename_list

	filename_list = list(
		itertools.chain.from_iterable(
			map(
				lambda dir_name:\
					map(\
						lambda fn: dir_name + fn,
						os.listdir( dir_name )
					),
				UNANSWERED_PATH_TUPLE
			)
		)
	)

	print(len(filename_list))

	define_global_tags()

	main_tm_dict = dict()

	min_options = max_options = None

	for file_tm_dict, file_min_options, file_max_options in Pool(n_proc).map(get_stats_per_proc,range(len(filename_list))):

		for tm, stats_dict in file_tm_dict.items():
			if tm in main_tm_dict:
				main_tm_dict[ tm ][ q_count_tag ] += stats_dict[ q_count_tag ]
				if stats_dict[ min_read_tag ] < main_tm_dict[ tm ][ min_read_tag ]:
					main_tm_dict[ tm ][ min_read_tag ] = stats_dict[ min_read_tag ]
				if stats_dict[ max_read_tag ] < main_tm_dict[ tm ][ max_read_tag ]:
					main_tm_dict[ tm ][ max_read_tag ] = stats_dict[ max_read_tag ]
				main_tm_dict[ tm ][ total_read_tag ] += stats_dict[ total_read_tag ]
			else:
				main_tm_dict[ tm ] = stats_dict

		if file_min_options is not None\
			and ( min_options is None or file_min_options < min_options ):
			min_options = file_min_options

		if file_max_options is not None\
			and ( max_options is None or file_max_options > max_options ):
			max_options = file_max_options

	for stats_dict in main_tm_dict.values():
		stats_dict[avg_read_tag] = stats_dict[total_read_tag] / stats_dict[q_count_tag]

	pickle.dump(
		{
			tm_stats_dict_tag : main_tm_dict,
			min_options_count_tag : min_options,
			max_options_count_tag : max_options,
		},
		open(
			'stats_dict.p',
			'wb'
		)
	)

def get_info_out_of_stats_dict():
	'''
	Reads and shows statistics from analyse_shape_of_input_data_main.
	'''
	stats_dict = pickle.load(open('stats_dict.p','rb'))

	define_global_tags()

	print('number of time tags: '\
		+ str(len(stats_dict[tm_stats_dict_tag].keys())))

	print('min number of q per time moment: '\
		+ str(
			min(
				map(
					lambda value: value[q_count_tag],
					stats_dict[tm_stats_dict_tag].values()
				)
			)
		))

	print('max number of q per time moment: '\
		+ str(
			max(
				map(
					lambda value: value[q_count_tag],
					stats_dict[tm_stats_dict_tag].values()
				)
			)
		))

	count_dict = dict()

	for value in stats_dict[tm_stats_dict_tag].values():

		if value[q_count_tag] in count_dict:
			count_dict[value[q_count_tag]] += 1
		else:
			count_dict[value[q_count_tag]] = 1

	a_list = sorted( count_dict.items() )

	print('most frequent # q per time moment: '
		+ str(max(a_list,key=lambda e: e[1])[0]))

	plt.plot(
		list(map(lambda e: e[0], a_list)),
		list(map(lambda e: e[1], a_list))
	)
	plt.show()

def get_week_times():
	'''
	Shows time intervals for the 3 weeks of grid activity.
	'''
	def get_min_max_time(folder):
		i = 0
		while i < 201:

			a_list = json.load(
				open(
					folder + str(i) + '.json',
					'rt'
				)
			)

			if len(a_list) > 0:

				min_time = a_list[0][0]

				break

			i += 1

		i = 200
		while i >= 0:

			a_list = json.load(
				open(
					folder + str(i) + '.json',
					'rt'
				)
			)

			if len(a_list) > 0:
				return (min_time, a_list[-1][0],)

			i-=1

	print('week 0 is between: ' + str(get_min_max_time(UNANSWERED_PATH_TUPLE[0])))
	print('week 1 is between: ' + str(get_min_max_time(UNANSWERED_PATH_TUPLE[1])))
	print('week 2 is between: ' + str(get_min_max_time(UNANSWERED_PATH_TUPLE[2])))

def get_per_proc_stats(i):
	q_count_sum = 0
	time_tag_count = 0
	for tm, q_count in particular_q_count_list:
		if particular_thp_list[i] - time_interval_length <= tm < particular_thp_list[i]:
			q_count_sum += q_count
			time_tag_count += 1
		elif particular_thp_list[i] <= tm:
			break
	return q_count_sum, time_tag_count

def get_stats_per_throughput_csv():
	'''
	Gathers statistics.
	'''
	stats_dict = pickle.load(open('stats_dict.p','rb'))

	define_global_tags()

	q_time_moments_list = sorted(
		map(
			lambda p: (p[0], p[1][q_count_tag],),
			stats_dict[tm_stats_dict_tag].items()
		)
	)

	limits_list = list(map(lambda e: e * 60 * 1000, range(1,61)))

	# min_q_count
	# 	minimum query count across all time moments
	# max_q_count
	#	maximum query count across all time moments
	# total_q_count
	#	total query count in a one minute interval
	# min_tm_count
	#	minimum number of time moments during one minute ; it is always 60
	# max_tm_count
	#	maximum number of time moments during one minute ; it is always 60
	# thp_count
	#	throughput points reported in one minute. The system does not log a new
	#	value of throughput if no change happens.
	Stats_Element = namedtuple('Stats_El', ['min_q_count', 'max_q_count', 'total_q_count',\
		'min_tm_count' , 'total_tm_count' , 'max_tm_count','thp_count'])

	def get_thp_info_per_week(first_tm, last_tm, thp_csv_file):

		thp_iterable = csv.reader(open(thp_csv_file,'rt'))

		next(thp_iterable)

		thp_list = list(
			filter(
				lambda e: first_tm + 60000 <= e < last_tm,
				map(
					lambda p: 1000 * int(p[0]),
					filter(
						lambda line_list: len(line_list) == 2,
						thp_iterable
					)
				)
			)
		)

		time_window_stats_list = list()

		global particular_thp_list, particular_q_count_list, time_interval_length

		for l in limits_list:

			particular_thp_list = list(
				filter(
					lambda e: first_tm + l <= e < last_tm,
					thp_list
				)
			)

			particular_q_count_list = list(
				filter(
					lambda p: first_tm <= p[0] <= last_tm,
					q_time_moments_list
				)
			)

			time_interval_length = l

			p = Pool(n_proc)

			q_count_per_thp_list = p.map(get_per_proc_stats, range(len(particular_thp_list)))

			p.close()

			time_window_stats_list.append(
				Stats_Element(
					min_q_count=min(q_count_per_thp_list,key=lambda e: e[0])[0],
					max_q_count=max(q_count_per_thp_list,key=lambda e: e[0])[0],
					total_q_count=sum(map(lambda e: e[0],q_count_per_thp_list)),
					min_tm_count=min(q_count_per_thp_list,key=lambda e: e[1])[1],
					total_tm_count=sum(map(lambda e: e[1],q_count_per_thp_list)),
					max_tm_count=max(q_count_per_thp_list,key=lambda e: e[1])[1],
					thp_count=len(q_count_per_thp_list),
				)
			)

		return time_window_stats_list

	min_q_count_list, max_q_count_list, avg_q_count_list,\
	min_tm_count_list, max_tm_count_list, avg_tm_count_list = list(),list(),list(),list(),list(),list(),

	for s_el_0, s_el_1, s_el_2 in zip(
		get_thp_info_per_week(WEEK_TIME_MOMENTS[0][0],WEEK_TIME_MOMENTS[0][1],ASSIGNED_THROUGHPUT_CSV_FILES[0]),
		get_thp_info_per_week(WEEK_TIME_MOMENTS[1][0],WEEK_TIME_MOMENTS[1][1],ASSIGNED_THROUGHPUT_CSV_FILES[1]),
		get_thp_info_per_week(WEEK_TIME_MOMENTS[2][0],WEEK_TIME_MOMENTS[2][1],ASSIGNED_THROUGHPUT_CSV_FILES[2]),
		):
		min_q_count_list.append(min(
			(s_el_0.min_q_count, s_el_1.min_q_count, s_el_2.min_q_count,)
		))
		max_q_count_list.append(max(
			(s_el_0.max_q_count, s_el_1.max_q_count, s_el_2.max_q_count,)
		))
		avg_q_count_list.append(
			(s_el_0.total_q_count + s_el_1.total_q_count + s_el_2.total_q_count)\
			/(s_el_0.thp_count + s_el_1.thp_count + s_el_2.thp_count)
		)
		min_tm_count_list.append(min(
			(s_el_0.min_tm_count, s_el_1.min_tm_count, s_el_2.min_tm_count,)
		))
		max_tm_count_list.append(max(
			(s_el_0.max_tm_count, s_el_1.max_tm_count, s_el_2.max_tm_count,)
		))
		avg_tm_count_list.append(
			(s_el_0.total_tm_count + s_el_1.total_tm_count + s_el_2.total_tm_count)\
			/(s_el_0.thp_count + s_el_1.thp_count + s_el_2.thp_count)
		)

	pickle.dump(
		(
			limits_list,
			min_q_count_list, max_q_count_list, avg_q_count_list,\
			min_tm_count_list, max_tm_count_list, avg_tm_count_list
		),
		open(
			'pipe.p',
			'wb'
		)
	)

def get_options_count_numbers_per_proc(i):
	res_list = 9 * [0]

	for q_list in json.load(open(filename_list[i],'rt')):
		res_list[len(q_list[2])-1] += 1

	return res_list

def get_option_counts():
	'''
	Get min and max number of options per query.
	Minimum was 1.
	Maximum was 9 !
	'''
	global filename_list
	filename_list = list(
		itertools.chain.from_iterable(
			map(
				lambda dir_name:\
					map(\
						lambda fn: dir_name + fn,
						os.listdir( dir_name )
					),
				UNANSWERED_PATH_TUPLE
			)
		)
	)

	counts_per_options_list = 9 * [0]

	for a_list in Pool(n_proc).map(get_options_count_numbers_per_proc, range(len(filename_list))):
		counts_per_options_list = map(
			lambda p: p[0] + p[1],
			zip( a_list , counts_per_options_list )
		)

	counts_per_options_list = list(counts_per_options_list)

	counts_sum = sum(counts_per_options_list)

	i = 1

	while i <= 9:

		print(str(i) + ' : ' + str(100*counts_per_options_list[i-1]/counts_sum))

		i+=1

def get_set_per_proc(i):
	return set(
		itertools.chain.from_iterable(
			map(
				lambda q_list: map(lambda p: p[0], q_list[2]),
				json.load(open(filename_list[i],'rt'))
			)
		)
	)

def get_set_of_ses():
	'''
	Get set of all storage elements.
	'''
	global filename_list
	filename_list = list(
		itertools.chain.from_iterable(
			map(
				lambda dir_name:\
					map(\
						lambda fn: dir_name + fn,
						os.listdir( dir_name )
					),
				UNANSWERED_PATH_TUPLE
			)
		)
	)

	a_set = set()
	for s in Pool(n_proc).map(get_set_per_proc,range(len(filename_list))):
		a_set.update(s)

	print(len(a_set))
	print(a_set)

# {'spbsu', 'kisti_gsdc', 'grenoble', 'kolkata', 'rrc-ki', 'ornl', 'strasbourg_ires', 'cyfronet', 'subatech', 'saopaulo', 'rrc_ki_t1', 'kfki', 'ccin2p3', 'jinr', 'sarfti', 'unam_t1', 'bratislava', 'troitsk', 'ihep', 'mephi', 'legnaro', 'itep', 'hiroshima', 'niham', 'icm', 'upb', 'pnpi', 'ipnl', 'fzk', 'catania', 'za_chpc', 'ral', 'kosice', 'cnaf', 'grif_ipno', 'iss', 'ndgf', 'bari', 'poznan', 'birmingham', 'bitp', 'nipne', 'trieste', 'gsi', 'cern', 'grif_irfu', 'prague', 'lbl_hpcs', 'torino', 'snic', 'sut', 'sara', 'clermont'}

if __name__ == '__main__':
	global n_proc

	n_proc = 95

	get_set_of_ses()