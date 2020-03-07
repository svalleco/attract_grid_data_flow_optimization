import pickle
import csv
from multiprocessing import Pool, Manager, Process, Lock
from multiprocessing.sharedctypes import RawArray
import os
import json
from functools import reduce
from collections import namedtuple
from copy import deepcopy

def bins_per_proc_5(ii):
	Bin_Element = namedtuple('Bin_El',['ind','fi_t','la_t','thp_t','bin_list','thp_v', 'thp_i', 'ipc_elem'])

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

	q_index = 0
	for time_stamp, read_size in g_query_list[ slice_start : slice_end ]:
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
							deepcopy(initial_bin_list),
							g_thp_list[thp_i][1],
							thp_i,
							g_ipc_list[thp_i],
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
							queue_list[q_i].thp_i,
							queue_list[q_i].ipc_elem,
						)

						break

					bin_i += 1

					bin_t += g_bin_length_in_time

		for bin_el in queue_list:

			if bin_el.ipc_elem.lock is not None:
				bin_el.ipc_elem.lock.acquire()

			bin_el.ipc_elem.array[bin_el.ind] += read_size

			if bin_el.ipc_elem.lock is not None:
				bin_el.ipc_elem.lock.release()

		q_index += 1

def get_five_minute_binned_dataset_5(
	p_query_iterable,
	p_throughput_iterable,
	result_path,
	first_moment,
	last_moment,
	n_proc,
	millis_interval_start=4000000,
	millis_interval_end=0,
	number_of_bins_per_thp=1000,):

	global g_thp_list, g_query_list, g_slice_list,\
		g_number_of_bins_per_thp, g_bin_length_in_time, g_millis_interval_start,\
		g_ipc_list

	g_query_list = p_query_iterable
	g_thp_list = p_throughput_iterable

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

	IPC_Element = namedtuple('IPC_El',['array', 'lock'])

	g_ipc_list = list()
	for tm, _ in g_thp_list:

		is_race_condition_flag = False
		for _, s_end in g_slice_list[:-1]:
			if tm - millis_interval_start <= g_query_list[s_end][0] < tm:
				is_race_condition_flag = True
				break

		g_ipc_list.append(
			IPC_Element(
				array=RawArray('d',number_of_bins_per_thp*[0,]),
				lock=Lock() if is_race_condition_flag else None,
			)
		)

	g_number_of_bins_per_thp = number_of_bins_per_thp

	g_bin_length_in_time = (millis_interval_start - millis_interval_end) / number_of_bins_per_thp

	g_millis_interval_start = millis_interval_start

	p = Pool(n_proc)

	print('Will start pool !')

	p.map(
		bins_per_proc_5,
		range(n_proc)
	)

	p.close()

	del g_query_list
	del g_slice_list

	json.dump(
		tuple(
			map(
				lambda ind: tuple(g_ipc_list[ind].array) + (g_thp_list[ind][1],),
				range(len(g_thp_list))
			)
		),
		open(result_path,'wt')
	)

	return p