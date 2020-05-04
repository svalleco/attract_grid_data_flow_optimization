import csv

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

def create_data_set(thp_iterable, read_size_iterable, distance_iterable, demotion_iterable):

	def reduce_throughput_at_front(thp_iterable_0, a_iterable, time_margin=0):
		thp_i = 0

		while thp_iterable_0[thp_i][0] - time_margin < a_iterable[0][0]: thp_i += 1

		return thp_iterable_0[thp_i:]

	thp_iterable = reduce_throughput_at_front(thp_iterable, read_size_iterable, 120000)

	thp_iterable = reduce_throughput_at_front(thp_iterable, distance_iterable)

	thp_iterable = reduce_throughput_at_front(thp_iterable, demotion_iterable)

	def reduce_throughput_at_back(thp_iterable_0, a_iterable):
		thp_i = len( thp_iterable_0 ) - 1

		while thp_iterable_0[thp_i][0] > a_iterable[-1][0]: thp_i -= 1

		return thp_iterable_0[:thp_i + 1]

	thp_iterable = reduce_throughput_at_back(thp_iterable, read_size_iterable)

	thp_iterable = reduce_throughput_at_back(thp_iterable, distance_iterable)

	thp_iterable = reduce_throughput_at_back(thp_iterable, demotion_iterable)

	data_set_list = list()

	thp_i = 0

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
				time_window_rs / 120,
				distance_iterable[dist_i][1],
				demotion_iterable[dem_i][1],
			)
		)

		thp_i += 1

	return data_set_list

def get_clients_and_ses_minimal_list_1(whole_matrix_per_week_list):

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