from multiprocessing import Pool, Lock
from multiprocessing.sharedctypes import RawArray
import json
import pickle
from collections import namedtuple

def analyse_main():
	g_query_list = sorted(json.load(open('first_opt_cern_only_read_value.json', 'rt')))

	g_thp_list = sorted(pickle.load(open('thp_dump_list.p','rb')))

	print()

def sequencial_bin_main(first_moment,
	millis_interval_start=4000000,
	number_of_bins_per_thp=1000):
	g_query_list = sorted(json.load(open('first_opt_cern_only_read_value.json', 'rt')))

	g_thp_list = sorted(pickle.load(open('thp_dump_list.p','rb')))

	print('Finished read !')

	i = 0
	while g_query_list[i][0] < first_moment:
		i += 1
	g_query_list = g_query_list[i:]

	if g_thp_list[0][0] - millis_interval_start < g_query_list[0][0]:
		i = 0
		while g_thp_list[i][0] - millis_interval_start < g_query_list[0][0]:
			i += 1
		g_thp_list = g_thp_list[i:]

	i = 0
	while g_query_list[i][0] < g_thp_list[0][0] - millis_interval_start:
		i += 1
	g_query_list = g_query_list[i:]

	if g_query_list[-1][0] <= g_thp_list[-1][0]:
		i = len(g_thp_list) - 1
		while g_query_list[-1][0] <= g_thp_list[i][0]:
			i -= 1
		g_thp_list = g_thp_list[:i+1]

	i = len(g_query_list) - 1
	while g_query_list[i][0] >= g_thp_list[-1][0]:
		i-=1
	g_query_list = g_query_list[:i+1]

	# print((g_thp_list[0][0] - millis_interval_start) / 60000)
	# print(g_query_list[0][0] / 60000)
	# print(g_query_list[-1][0] / 60000)
	# print(g_thp_list[-1][0] / 60000)

	result_list = []
	for _ in range(len(g_thp_list)):
		a = []
		for _ in range(number_of_bins_per_thp):
			a.append(0)
		result_list.append(a)

	g_bin_length_in_time = millis_interval_start / number_of_bins_per_thp

	thp_i = 0

	Bin_Element = namedtuple('Bin_El',['bin_ind','fi_t','la_t','thp_t','bin_list','thp_v', 'thp_i'])

	queue_list = []

	a_i = 0

	for time_stamp, read_size in g_query_list:

		if a_i % 100 == 0:
			print(a_i)

		fl = True

		new_queue_list = []
		for bin_el in queue_list:
			if bin_el.thp_t <= time_stamp:
				result_list[ bin_el.thp_i ] = list(
					map(
						lambda p: p[0] + p[1],
						zip(
							result_list[ bin_el.thp_i ],
							bin_el.bin_list
						)
					)
				)
				fl = False
				break
			else:
				new_queue_list.append(bin_el)
		queue_list = new_queue_list

		if not fl: break

		thp_i_0 = thp_i
		while thp_i_0 < len(g_thp_list) and g_thp_list[thp_i_0][0] - g_thp_list[thp_i][0] < 10 * 60 * 1000:

			if g_thp_list[thp_i_0][0] - millis_interval_start <= time_stamp < g_thp_list[thp_i_0][0]:
				bin_i = 0

				bin_t = g_thp_list[thp_i_0][0] - millis_interval_start

				while True:
					if bin_t <= time_stamp < bin_t + g_bin_length_in_time:

						queue_list.append(
							Bin_Element(
								bin_i,
								bin_t,
								bin_t + g_bin_length_in_time,
								g_thp_list[thp_i_0][0],
								[0 for _ in range(number_of_bins_per_thp)],
								g_thp_list[thp_i_0][1],
								thp_i_0,
							)
						)

						break

					bin_t += g_bin_length_in_time

					bin_i += 1

			thp_i_0 += 1

		thp_i = queue_list[-1].thp_i

		for q_i in range(len(queue_list)):
			if not ( queue_list[q_i].fi_t <= time_stamp < queue_list[q_i].la_t ):

				bin_i = queue_list[q_i].bin_ind

				bin_t = queue_list[q_i].fi_t

				while bin_i < number_of_bins_per_thp:

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
			bin_el.bin_list[bin_el.bin_ind] += read_size

		a_i += 1

	for bin_el in queue_list:
		result_list[ bin_el.thp_i ] = list(
			map(
				lambda p: p[0] + p[1],
				zip(
					result_list[ bin_el.thp_i ],
					bin_el.bin_list
				)
			)
		)

	plm = tuple(
			map(
				lambda ind: tuple(result_list[ind]) + (g_thp_list[ind][1],),
				range(len(g_thp_list))
			)
		)

	print(plm[0])

	json.dump(
		plm,
		open('data_set.json','wt')
	)

if __name__ == '__main__':
	first_moment, last_moment = 1579264801390, 1579875041000

	if True:
		sequencial_bin_main(first_moment)