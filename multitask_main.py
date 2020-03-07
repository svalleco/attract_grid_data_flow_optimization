import csv
import os
from multiprocessing import Pool, Manager
import pickle
from functools import reduce
import pickle
# import numpy as np
from scipy import interpolate
from functools import reduce
import random
import matplotlib.pyplot as plt
# from statsmodels.tsa.seasonal import seasonal_decompose
import csv

def analyse_thp_0():
	thp_list = pickle.load(open('cern_throughput_per_site_list.p','rb'))

	min_clients = 2000
	max_clients = -1
	avg_clients = 0
	count = 0


	for _, ts_dict in thp_list:
		if len(ts_dict) < min_clients: min_clients = len(ts_dict)
		if len(ts_dict) > max_clients: max_clients = len(ts_dict)
		avg_clients += len(ts_dict)
		count += 1

	print(min_clients)
	print(avg_clients/count)
	print(max_clients)


	for _, ts_dict in thp_list:
		if len(ts_dict) == min_clients:

			print(sorted(ts_dict.keys()))

			break

	for _, ts_dict in thp_list:
		if len(ts_dict) == max_clients:

			print(sorted(ts_dict.keys()))

			break

	min_count = 0
	max_count = 0

	for _, ts_dict in thp_list:
		if len(ts_dict) == max_clients:
			max_count += 1
		if len(ts_dict) == min_clients: min_count += 1

	print(min_count)
	print(max_count)

def analyse_thp_1():
	cern_rs_list = pickle.load(open('pipe_3.p','rb'))

	min_v, max_v =\
		min(map(lambda e: e[1], cern_rs_list)),\
		max(map(lambda e: e[1], cern_rs_list))

	a_rs_tuple = tuple(map(lambda e: e[0], cern_rs_list))
	b_rs_tuple = tuple(map(lambda e: (e[1]-min_v)/(max_v-min_v), cern_rs_list))

	del cern_rs_list

	thp_list = pickle.load(open('cern_throughput_per_site_list.p','rb'))

	a_list = []
	b_list = []

	for ts, ts_dict in thp_list:
		a_list.append(ts)
		b_list.append(ts_dict['cern'])

	del thp_list

	min_v, max_v =\
		min(map(lambda e: e, b_list)),\
		max(map(lambda e: e, b_list))

	b_list = tuple(map(lambda e: (e-min_v)/(max_v-min_v), b_list))

	min_v = min((
		min( a_rs_tuple ),
		min( a_list )
	))

	plt.plot(
		tuple(
			map(
				lambda e: (e - min_v) / (60*1000),
				a_rs_tuple
			)
		),
		b_rs_tuple,
		# 'b+'
	)

	plt.plot(
		tuple(
			map(
				lambda e: (e - min_v) / (60*1000),
				a_list
			)
		),
		b_list,
		# 'r+'
	)

	plt.show()

def analyse_q_emitted_by_cern_0():
	q_dict = dict()
	for ts, thp in map(
			lambda e: (e[0],e[-1],),
			filter(
				lambda l: 'cern' in l[1] and 'cern' in l[2][0],
				pickle.load(open('answered_cern_queries.p','rb'))
			)
		):
		if ts in q_dict:
			q_dict[ts] += thp
		else:
			q_dict[ts] = thp

	pickle.dump(
		sorted(q_dict.items()),
		open('pipe_3.p','wb')
	)

def analyse_q_emitted_by_cern_1():
	s = set()
	for first_option in map(
		lambda e: e[2][0],
		pickle.load(open('answered_cern_queries.p','rb'))
	):
		s.add(first_option)

	for se in s:
		print(se)

def get_thp_per_proc_2(i):
	g_return_list.extend(
		tuple(
			filter(
				lambda p: g_first_moment + 300000 <= p[0] <= g_last_moment,
				map(
					lambda e: (int(e[1]), e[5].split('_OUT')[0], float(e[6]),),
					filter(
						lambda e: len(e) == 7\
							and g_client_name in e[2].lower()\
							and '_OUT_freq' not in e[5]\
							and '_IN' not in e[5],
						map(
							lambda line: line.split('\t'),
							open('../remote_host/spool/'+filename_list[i],'rt').read().split('\n')
						)
					)
				)
			)
		)
	)

def get_throughput_2(first_moment, last_moment, client_name):
	spool_dir_path = '../remote_host/spool/'

	global filename_list, g_client_name, g_first_moment, g_last_moment,\
		g_return_list

	g_first_moment, g_last_moment, g_client_name =\
		first_moment, last_moment, client_name

	if False:
		filename_list = list(
			filter(
				lambda e:\
					'.done' in e\
					and first_moment <= int(e[:-5]) < last_moment + 3600000,
				os.listdir(spool_dir_path)
			)
		)
	if True:
		filename_list = list(
			filter(
				lambda e:\
					'.done' in e\
					and first_moment - 3600000 <= int(e[:-5]) < last_moment + 3600000,
				os.listdir(spool_dir_path)
			)
		)

	g_return_list = Manager().list()

	p = Pool(n_proc)

	p.map(get_thp_per_proc_2, range(len(filename_list)))

	p.close()

	return_dict = dict()

	for time_stamp, from_name, value in g_return_list:
		if from_name not in return_dict:
			return_dict[from_name] = [(time_stamp,value,),]
		else:
			return_dict[from_name].append((time_stamp,value,))

	for key in return_dict.keys():
		return_dict[key].sort()

	pickle.dump(
		return_dict,
		open(
			'cern_site_thp_list_dict.p',
			'wb'
		)
	)

def get_thp_dump_based_on_interp(first_moment,last_moment):
	interp_func_dict = list()
	for from_name,thp_list in pickle.load(open('cern_site_thp_list_dict.p','rb')).items():
		interp_func_dict.append(
			(
				from_name,
				interpolate.interp1d(
					list(map(lambda p: p[0], thp_list)),
					list(map(lambda p: p[1], thp_list)),
				),
			)
		)

	thp_dump_list = list()

	time_i = first_moment
	while time_i <= last_moment:
		thp_val = 0
		for from_name, f in interp_func_dict:
			try:
				thp_val += f(time_i)
			except:
				pass
		thp_dump_list.append((time_i,thp_val,))
		time_i += 60000

	pickle.dump(
		thp_dump_list,
		open('thp_dump_list.p','wb')
	)

def analyse_thp_2():
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

	min_a, max_a = min(y_list), max(y_list)

	y_list = list(
		map(
			lambda e: (e - min_a) / (max_a - min_a), y_list
		)
	)

	a_dict = dict()
	for q_time, q_read_size in pickle.load(open('queries_throughput_list.p', 'rb')):
		if q_time in a_dict:
			a_dict[q_time] += q_read_size
		else:
			a_dict[q_time] = q_read_size

	key_list = sorted( a_dict.keys() )

	min_a, max_a =\
		min(map(lambda k: a_dict[k], key_list)),\
		max(map(lambda k: a_dict[k], key_list))

	rs_list = tuple(
		map(lambda k: ( a_dict[k] - min_a ) / ( max_a - min_a ), key_list)
	)

	thp_dump_list = pickle.load(open('thp_dump_list.p', 'rb'))

	min_a, max_a =\
		min(map(lambda k: k[1], thp_dump_list)),\
		max(map(lambda k: k[1], thp_dump_list))

	pickle.dump(
		(
			key_list,
			rs_list,
			tuple(map(lambda e: e[0], thp_dump_list)),
			tuple(map(lambda e: ( e[1] - min_a ) / ( max_a - min_a ), thp_dump_list)),
			x_list,
			y_list,

		),
		open('pipe_1.p','wb')
	)

def write_to_csv(filename, x_iterable, y_iterable, x_name, y_name):
	file_handle = open(filename,'wt')

	file_handle.write( x_name + ',' + y_name + '\n' )

	for x, y in zip(x_iterable,y_iterable):
		file_handle.write( str(x) + ',' + str(y) + '\n' )

def analyse_thp_3():
	a,b,c,d, _,_ = pickle.load(open('pipe_1.p','rb'))

	a=tuple(map(lambda e: (e - 1576450800000) / 60000, a))

	c=tuple(map(lambda e: (e - 1576450800000) / 60000, c))

	thp_list = pickle.load(open('throughput_dump.p','rb'))

	min_a,max_a=\
		min(map(lambda e: e[1],thp_list)),\
		max(map(lambda e: e[1],thp_list))

	e,f=\
		tuple( map( lambda e: ( e[0] - 1576450800000 ) / 60000 , thp_list ) ),\
		tuple( map( lambda e: ( e[1] - min_a ) / ( max_a - min_a ) , thp_list ) ),

	g,h = reduce(
		lambda acc, x: (\
			acc[0] + ( ( x[0] - 1576450800 ) / 60 , ),\
			acc[1] + ( x[1] , ),\
		),
		filter(
			lambda e: 1576450800000 <= e[0] * 1000 <= 1576537200000,
			map(
				lambda line: ( int( line[0] ) , float( line[1] ) , ) ,
				csv.reader(open('from_web_thp.csv'),delimiter=',')
			)
		),
		(tuple(),tuple(),)
	)

	min_a,max_a= min(h), max(h)

	h = tuple(map(lambda e: ( e - min_a ) / ( max_a - min_a ) , h))

	# write_to_csv(
	# 	'read_size.csv',
	# 	a,
	# 	b,
	# 	'time_in_minutes',
	# 	'normalized_value',
	# )
	# write_to_csv(
	# 	'old_throughput.csv',
	# 	e,
	# 	f,
	# 	'time_in_minutes',
	# 	'normalized_value',
	# )
	# write_to_csv(
	# 	'new_throughput.csv',
	# 	c,
	# 	d,
	# 	'time_in_minutes',
	# 	'normalized_value',
	# )
	# write_to_csv(
	# 	'throughput_from_site.csv',
	# 	g,
	# 	h,
	# 	'time_in_minutes',
	# 	'normalized_value',
	# )

	plt.plot(a,b,label='read size')
	plt.plot(e,f,label='old throughput')
	plt.plot(c,d,label='new throughput')
	plt.plot(g,h,label='throughput from site')
	plt.xlabel('Time in minutes')
	plt.ylabel('Normalized values')
	plt.legend()
	plt.show()

def analyse_thp_4():
	x_list, y_list = list(), list()

	thp_dump_list = pickle.load(open('thp_dump_list.p', 'rb'))

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

	x_list = tuple( map( lambda e: (e - min(x_list)) / 60000, x_list) )
	min_a,max_a = min(y_list),max(y_list)
	print((min_a,max_a))
	y_list = tuple( map( lambda e: (e - min_a)/(max_a-min_a), y_list) )

	_,_,_,_,a,b = pickle.load(open('pipe_1.p','rb'))


	# plt.plot(
	# 	tuple(map(lambda e: (e - 1576450800000) / 60000,a)),
	# 	b,
	# 	label='read size'
	# )
	plt.plot(
		x_list,
		y_list,
		label='new throughput'
	)

	write_to_csv(
		'third_feb_read_size.csv',
		tuple(map(lambda e: (e - 1576450800000) / 60000,a)),
		b,
		'time_in_minutes',
		'normalized_value',
	)
	write_to_csv(
		'third_feb_old_throughput.csv',
		x_list,
		y_list,
		'time_in_minutes',
		'normalized_value',
	)

	plt.xlabel('Time in minutes')
	plt.ylabel('Normalized values')
	plt.legend()
	plt.show()

def plot_throughputs():
	import matplotlib
	def extract_lists(filename):
		old_generator = csv.reader(open(filename,'rt'))
		next(old_generator)
		old_x_list, old_y_list = list(), list()
		for x, y in old_generator:
			old_x_list.append(float(x))
			old_y_list.append(float(y))
		return old_x_list, old_y_list

	old_x_list, old_y_list = extract_lists('old_throughput.csv')

	new_x_list, new_y_list = extract_lists('new_throughput.csv')

	site_x_list, site_y_list = extract_lists('throughput_from_site.csv')

	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)
	plt.plot(old_x_list,old_y_list,label='old throughput')
	plt.plot(new_x_list,new_y_list,label='new throughput')
	plt.plot(site_x_list,site_y_list,label='throughput from site')
	# plt.rcParams.update({'font.size': 22})
	# matplotlib.rc('xtick', labelsize=20)
	# matplotlib.rc('ytick', labelsize=20)
	plt.xlabel('Time in minutes')
	plt.ylabel('Normalized values')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	if True:
		#one week
		first_moment, last_moment = 1579264801390, 1579875041000
	if False:
		first_moment, last_moment = 1576450800000, 1576537200000

	global n_proc
	n_proc = 7

	if False:
		get_throughput_2(first_moment, last_moment, 'cern')
	if False:
		get_thp_dump_based_on_interp(first_moment,last_moment)
	if False:
		analyse_thp_2()
	if True:
		plot_throughputs()
