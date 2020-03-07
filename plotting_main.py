import matplotlib.pyplot as plt
import pickle
import json
import csv
from functools import reduce
import os
import numpy as np
from scipy import fftpack
from scipy import interpolate
from multiprocessing import Pool
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import kurtosis
from stldecompose import decompose
from statsmodels.tsa.x13 import x13_arima_analysis
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

def plot_main_0():
	ground_truth_list, predicted_list = pickle.load(open(
		'from_notebook/to_transfer/best_seasonal_trend_noise.p','rb'
	))

	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	import matplotlib.ticker as ticker
	ax = plt.gca()
	ax.xaxis.set_major_locator(ticker.MultipleLocator(720))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

	plt.plot(
		range( len( ground_truth_list ) ),
		tuple( map( lambda e: e[2] , ground_truth_list ) ),
		label='Ground Truth Noise'
	)

	plt.plot(
		range( len( predicted_list ) ),
		tuple( map( lambda e: e[2] , predicted_list ) ),
		label='Predicted Noise'
	)

	time = 0
	while time < len( predicted_list ):
		plt.plot(
			(time,time),
			(0,1),
			'r-'
		)
		time+=720

	plt.xlabel('Index in The Data Set')
	plt.ylabel('Normalized Value')
	plt.legend()
	plt.show()

def plot_main_1():
	ground_truth_list, predicted_list = pickle.load(open(
		'from_notebook/to_transfer/best_site_thp.p','rb'
	))

	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	import matplotlib.ticker as ticker
	ax = plt.gca()
	ax.xaxis.set_major_locator(ticker.MultipleLocator(720))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

	plt.plot(
		range( len( ground_truth_list ) ),
		ground_truth_list,
		'b+',
		label='Ground Truth Throughput',
	)

	plt.plot(
		range( len( predicted_list ) ),
		predicted_list,
		'r+',
		label='Predicted Throughput',
	)

	time = 0
	while time < len( predicted_list ):
		plt.plot(
			(time,time),
			(0,1),
			'r-'
		)
		time+=720

	plt.xlabel('Index in The Data Set')
	plt.ylabel('Normalized Value')
	plt.legend()


	plt.show()

def plot_main_2():
	'''
	Plots reads size per time moment vs throughput
	'''
	if False:
		rs_dict = dict()
		for t, rs in json.load(open('first_opt_cern_only_read_value.json', 'rt')):
			if t in rs_dict:
				rs_dict[t] += rs
			else:
				rs_dict[t] = rs
		pickle.dump(
			sorted( rs_dict.items() ),
			open(
				'first_week_rs_per_time.p',
				'wb'
			)
		)
	a_list = pickle.load(open('first_week_rs_per_time.p','rb'))
	min_a, max_a = min(a_list,key=lambda e:e[1])[1],max(a_list,key=lambda e:e[1])[1]
	a_list = tuple(map(lambda e: (e[0],(e[1]-min_a)/(max_a-min_a),),a_list))

	thp_list =\
			tuple(
				filter(
					lambda p: a_list[0][0] <= p[0] <= a_list[-1][0],
					map(
						lambda p: (1000*int(p[0]), float(p[1]),),
						tuple(
							csv.reader(
								open('january_month_throughput.csv','rt')
							)
						)[1:]
					)
			)
		)
	min_a, max_a = min(thp_list,key=lambda e:e[1])[1],max(thp_list,key=lambda e:e[1])[1]
	thp_list = tuple(map(lambda e: (e[0],(e[1]-min_a)/(max_a-min_a),),thp_list))

	plt.plot(
		tuple(map(lambda e: e[0],a_list)), tuple(map(lambda e: e[1],a_list))
	)
	plt.plot(
		tuple(map(lambda e: e[0],thp_list)), tuple(map(lambda e: e[1],thp_list))
	)

	plt.show()

def plot_main_3():
	# min,max=(-34498.30027637715, 75772.12179454978)
	ground_truth_list,predicted_list=pickle.load(open('from_notebook/to_transfer/unitask_nn_trend.p','rb'))
	ground_truth_list = tuple(map(
		lambda e: (75772.12179454978 + 34498.30027637715) * e - 34498.30027637715,
		ground_truth_list
	))
	predicted_list = tuple(map(
		lambda e: (75772.12179454978 + 34498.30027637715) * e - 34498.30027637715,
		predicted_list
	))

	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	import matplotlib.ticker as ticker
	ax = plt.gca()
	ax.xaxis.set_major_locator(ticker.MultipleLocator(720))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

	plt.plot(
		range( len( ground_truth_list ) ),
		ground_truth_list,
		# 'b+',
		label='Ground Truth Trend',
	)

	plt.plot(
		range( len( predicted_list ) ),
		predicted_list,
		# 'r+',
		label='Predicted Trend',
	)

	time = 0
	while time < len( predicted_list ):
		plt.plot(
			(time,time),
			(
				min(ground_truth_list + predicted_list),
				max(ground_truth_list + predicted_list),
			),
			'r-'
		)
		time+=720

	plt.xlabel('Index in The Data Set')
	plt.ylabel('MB/s')
	plt.legend()

	plt.show()

def check_noise_main():
	ground_truth_list,predicted_list=pickle.load(open('from_notebook/to_transfer/unitask_nn_noise.p','rb'))
	ground_truth_list = tuple(map(
		lambda e: (75772.12179454978 + 34498.30027637715) * e - 34498.30027637715,
		ground_truth_list
	))

	min_noise, max_noise = min(ground_truth_list), max(ground_truth_list)

	granulation = 100

	step = (max_noise - min_noise) / granulation

	int_list = [ ( min_noise, min_noise + step ) ]
	while int_list[-1][1] - step < max_noise:
		int_list.append((
			int_list[-1][1],
			int_list[-1][1] + step
		))
	int_list.append((
		int_list[-1][1],
		max_noise + 1
	))

	a =\
	reduce(
		lambda acc,x: acc + ( ( x[0] , x[2] , ) , ( x[1] , x[2] , ) , ) ,
		map(
			lambda interval:\
			(
				interval[0],
				interval[1],
				len(
					tuple(filter(
						lambda e: interval[0] <= e < interval[1],
						ground_truth_list
					))
				),
			),
			int_list
		),
		tuple()
	)

	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	plt.plot(
		tuple(map(lambda e: e[0],a)),
		tuple(map(lambda e: e[1],a)),
		label='Noise Values Distribution'
	)

	mean_point = sum(ground_truth_list) / len(ground_truth_list)

	plt.plot(
		(mean_point, mean_point),
		( 0 , max(a,key=lambda e: e[1])[1], ),
		label='Noise Mean'
	)

	plt.xlabel('Noise Values')
	plt.ylabel('Number of Occurences in the Noise Set')
	plt.legend()

	plt.show()

def plot_matrices_gaps_main():
	first_moment, last_moment = 1579264801390, 1579875041000

	distance_fn_iterable =\
	sorted(
		filter(
			lambda e: e > last_moment,
			map(
				lambda fn: int(fn.split('_')[0]),
				filter(
					lambda fn: 'distance' in fn,
					os.listdir( '../remote_host/log_folder/' )
				)
			)
		)
	)

	prev_list = [distance_fn_iterable[0],]

	for fn_int in distance_fn_iterable[1:]:
		if fn_int - prev_list[-1] >= 21600000:
			prev_list.append(
				fn_int
			)

	min_a = min(prev_list)

	print(min_a)

	prev_list =\
	tuple(
		map(
			lambda e: (e - min_a) / (3600*1000),
			prev_list
		)
	)

	plt.plot(
		prev_list,
		len(prev_list) * [0],
		'b+'
	)

	t = 0
	while t < prev_list[-1]:

		plt.plot(
			(t, t,),
			(-1,1,),
			'r-'
		)

		print(t)

		t += 168

	plt.show()

	# 168
	# 336
	# 504

	# 1580488142233 -> 1581697742233

def plot_multi_client_data_for_validation():
	X = json.load(open('first_week_normalized_data_set_cern_all_clients.json','rt'))

	plt.plot(
		range(len(X)),
		tuple(map(lambda p: p[1][3],X)),
		'b-'
	)

	plt.show()

def plot_frequency_decomposition():

	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	thp_list =\
	tuple(
		map(
			lambda p: (int(p[0]), float(p[1]),),
			tuple(
				csv.reader(
					open('january_month_throughput.csv','rt')
				)
			)[1:]
		)
	)

	print(len(thp_list))

	f = interpolate.interp1d(
		tuple(map(lambda e: e[0], thp_list)),
		tuple(map(lambda e: e[1], thp_list))
	)

	min_t, max_t = min(map(lambda e: e[0], thp_list)), max(map(lambda e: e[0], thp_list))

	t = min_t

	x_list = list()

	while t <= max_t:

		x_list.append( f(t) )

		t+=120

	if False:
		pickle.dump(
			x_list,
			open(
				'two_minutes_spaced_throughput_data.p',
				'wb'
			)
		)

	X = fftpack.fft(x_list)

	freqs = fftpack.fftfreq(len(X)) * (1 / 120)

	plt.plot( freqs[:len(freqs)//2+1] , np.abs(X)[:len(freqs)//2+1] , 'b+')

	plt.xlabel('Frequecy in Hz')
	plt.ylabel('Amplitude of the Frequecy')

	plt.show()

def plot_tsa():
	thp_list = pickle.load(open('two_minutes_spaced_january_throughput_data.p','rb'))

	# result = seasonal_decompose(
	# 	thp_list,
	# 	model='additive',
	# 	freq=4260,
	# )

	result = decompose(thp_list, period=30)

	if False:
		'''
		Nu merge ca vrea quarterly sau monthly data.
		'''

		t = 1578936840000

		thp_dict = dict()

		for val in thp_list:
			thp_dict[pd.Timestamp(t)] = val
			t += 120000

		result = x13_arima_analysis(
			pd.Series(thp_dict,name="Thp"),
			x12path='/home/mircea/Downloads/x13asall_V1.1_B39/x13as'
		)

	preproc_for_plot_func=lambda arr:\
		tuple(\
			filter(\
				lambda e: str(e[1]) != 'nan',\
				enumerate(arr)\
			)\
		)

	a_func=lambda arr, ind: tuple(map(lambda p: p[ind],arr))

	trend_iterable = preproc_for_plot_func(result.trend)

	seasonal_iterable = preproc_for_plot_func(result.seasonal)

	resid_iterable = preproc_for_plot_func(result.resid)

	if False:

		plt.plot(range(len(thp_list)), thp_list, label='Original')

		plt.plot(a_func(trend_iterable,0), a_func(trend_iterable,1), label='Trend')

		plt.plot(a_func(seasonal_iterable,0), a_func(seasonal_iterable,1), label='Seasonal')

		plt.plot(a_func(resid_iterable,0), a_func(resid_iterable,1), label='Residual')

		plt.legend()

		plt.show()

	if True:
		import matplotlib
		font = {'family' : 'normal',
		        'weight' : 'bold',
		        'size'   : 22}
		matplotlib.rc('font', **font)
		import matplotlib.ticker as ticker

		b_func = lambda arr: tuple(\
			map( lambda p: ( 0.001388888888888889 * p[0] , p[1] , ) , arr ) )

		trend_iterable = b_func(trend_iterable)

		seasonal_iterable = b_func(seasonal_iterable)

		resid_iterable = b_func(resid_iterable)

		c_func = lambda arr, f: f(map(lambda e: e[1],arr))
		print('Original: [ ' + str(min(thp_list)) + ' , ' + str(max(thp_list)) + ' ] ')
		print('Trend: [ ' + str(c_func(trend_iterable,min)) + ' , ' + str(c_func(trend_iterable,max)) + ' ] ')
		print('Seasonal: [ ' + str(c_func(seasonal_iterable,min)) + ' , ' + str(c_func(seasonal_iterable,max)) + ' ] ')
		print('Residual: [ ' + str(c_func(resid_iterable,min)) + ' , ' + str(c_func(resid_iterable,max)) + ' ] ')

		if True:
			max_a = max(map(lambda e: 0.001388888888888889 * e, range(len(thp_list))))

		ax=plt.subplot(411)
		ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.041666666666666664))
		plt.plot(
			tuple(map(lambda e: 0.001388888888888889 * e, range(len(thp_list)))),
			thp_list,
			label='Throughput'
		)
		plt.plot(
			(0,0),
			(min(thp_list),max(thp_list)),
			'r-'
		)
		plt.plot(
			(max_a,max_a),
			(min(thp_list),max(thp_list)),
			'r-'
		)
		plt.plot()
		plt.ylabel('MB/s')
		plt.legend()

		ax=plt.subplot(412)
		ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.041666666666666664))
		plt.plot(
			a_func(trend_iterable,0),
			a_func(trend_iterable,1),
			label='Trend'
		)
		plt.plot(
			(0,0),
			(min(a_func(trend_iterable,1)),max(a_func(trend_iterable,1))),
			'r-'
		)
		plt.plot(
			(max_a,max_a),
			(min(a_func(trend_iterable,1)),max(a_func(trend_iterable,1))),
			'r-'
		)
		plt.ylabel('MB/s')
		plt.legend()

		ax=plt.subplot(413)
		ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.041666666666666664))
		plt.plot(
			a_func(seasonal_iterable,0),
			a_func(seasonal_iterable,1),
			label='Seasonal'
		)
		plt.plot(
			(0,0),
			(min(a_func(seasonal_iterable,1)),max(a_func(seasonal_iterable,1))),
			'r-'
		)
		plt.plot(
			(max_a,max_a),
			(min(a_func(seasonal_iterable,1)),max(a_func(seasonal_iterable,1))),
			'r-'
		)
		plt.ylabel('MB/s')
		plt.legend()

		ax=plt.subplot(414)
		ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.041666666666666664))
		plt.plot(
			a_func(resid_iterable,0),
			a_func(resid_iterable,1),
			label='Residual'
		)
		plt.plot(
			(0,0),
			(min(a_func(resid_iterable,1)),max(a_func(resid_iterable,1))),
			'r-'
		)
		plt.plot(
			(max_a,max_a),
			(min(a_func(resid_iterable,1)),max(a_func(resid_iterable,1))),
			'r-'
		)
		time = 0
		if False:
			while time - 1 < max_a:
				plt.plot(
					(time,time),
					(min(a_func(resid_iterable,1)),max(a_func(resid_iterable,1))),
					'r-'
				)
				time+=1
		plt.xlabel('Time in Days')
		plt.ylabel('MB/s')
		plt.legend()

		plt.show()

def plot_tsa_1():
	from statsmodels.tsa.seasonal import seasonal_decompose
	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 20}
	matplotlib.rc('font', **font)
	import matplotlib.ticker as ticker

	thp_list = pickle.load(open('two_minutes_spaced_january_throughput_data.p','rb'))[:1440]

	preproc_for_plot_func=lambda arr:\
		tuple(\
			map(\
				lambda p: (0.001388888888888889 * p[0], p[1],),\
				filter(\
					lambda e: str(e[1]) != 'nan',\
					enumerate(arr)\
				)\
			)\
		)

	a_func=lambda arr, ind: tuple(map(lambda p: p[ind],arr))

	trend_iterable =\
	preproc_for_plot_func(
		seasonal_decompose(
			thp_list,
			model='additive',
			freq=30
		).trend
	)
	ax=plt.subplot(311)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.041666666666666664))
	plt.plot(
		a_func(trend_iterable,0),
		a_func(trend_iterable,1),
		label='p = 1 hour'
	)
	plt.plot(
		tuple(map(lambda e: 0.001388888888888889 * e, range(len(thp_list)))),
		thp_list,
		label='original'
	)
	plt.ylabel('MB/s')
	plt.legend()

	trend_iterable =\
	preproc_for_plot_func(
		seasonal_decompose(
			thp_list,
			model='additive',
			freq=360
		).trend
	)
	ax=plt.subplot(312)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.041666666666666664))
	plt.plot(
		a_func(trend_iterable,0),
		a_func(trend_iterable,1),
		label='p = 12 hours'
	)
	plt.plot(
		tuple(map(lambda e: 0.001388888888888889 * e, range(len(thp_list)))),
		thp_list,
		label='original'
	)
	plt.ylabel('MB/s')
	plt.legend()

	trend_iterable =\
	preproc_for_plot_func(
		seasonal_decompose(
			thp_list,
			model='additive',
			freq=720
		).trend
	)
	ax=plt.subplot(313)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.041666666666666664))
	plt.plot(
		a_func(trend_iterable,0),
		a_func(trend_iterable,1),
		label='p = 24 hours'
	)
	plt.plot(
		tuple(map(lambda e: 0.001388888888888889 * e, range(len(thp_list)))),
		thp_list,
		label='original'
	)
	plt.xlabel('Time in Days')
	plt.ylabel('MB/s')
	plt.legend()

	plt.show()

def look_per_proc(p):
	k = kurtosis(
		tuple(map(
			lambda p: str(p) != 'nan',
			seasonal_decompose(
				thp_list,
				model='additive',
				freq=p
			).resid
		))
	)
	# k = kurtosis(
	# 	tuple(map(
	# 		lambda p: str(p) != 'nan',
	# 		decompose(thp_list, period=p).resid

	# 	))
	# )
	return (
		p,
		k,
		abs( 0 - k ),
	)

def look_for_best_noise():
	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 20}
	matplotlib.rc('font', **font)

	global thp_list
	thp_list = pickle.load(open('two_minutes_spaced_january_throughput_data.p','rb'))

	max_v = 5040
	# max_v = 50

	a = Pool(7).map(
		look_per_proc,
		range( 30 , max_v )
	)

	# print(a)

	print(
		min(
			a,
			key=lambda e: e[-1]
		)
	)

	print(
		max(
			a,
			key=lambda e: e[-1]
		)
	)

	plt.plot(
		range( 60 , 2 * max_v , 2),
		tuple(map(lambda p: p[1],a)),
		'bo'
	)

	# plt.plot(
	# 	range( 30 ,  max_v ),
	# 	tuple(map(lambda p: p[1],a)),
	# 	'bo'
	# )

	plt.xlabel('TSA period in minutes')
	plt.ylabel('Kurtosis')

	plt.show()

def plot_autocorrelation():
	thp_list = pickle.load(open('two_minutes_spaced_january_throughput_data.p','rb'))

	thp_mean = sum(thp_list) / len(thp_list)

	thp_centered_in_zero = tuple(
		map(
			lambda e: e - thp_mean,
			thp_list
		)
	)

	square_sum = sum(
		map(
			lambda e: e * e,
			thp_centered_in_zero
		)
	)

	print(len(thp_list))

	plt.plot(
		range(1, len(thp_list) - 1),
		tuple(
			map(
				lambda k:\
					sum(
						map(
							lambda p: p[0]*p[1],
							zip(
								thp_centered_in_zero[k:],
								thp_centered_in_zero[:len(thp_list)-k]
							)
						)
					) / square_sum,
				range(1, len(thp_list) - 1)
			)
		),
		'ro'
	)

	plot_acf(np.array(thp_list), lags=20158)
	# plot_pacf(np.array(thp_list), lags=50)
	plt.show()

if __name__ == '__main__':
	plot_autocorrelation()