import os
import matplotlib.pyplot as plt
from shutil import copyfile
import pickle
import itertools
from sklearn.decomposition import PCA
import numpy as np
from multiprocessing import Pool, Queue
import json
import csv
import random
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf
from collections import namedtuple
import itertools
from pca_utils import *
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf

'''
MATRIXES_FOLDER_LIST
	This is a tuple of folder local paths where the raw matrices files are located on skylake 07
	and Minsky NFS system.

WEEK_TIME_MOMENTS
	This is a tuple of time intervals for each week.

UNANSWERED_PATH_TUPLE
	This is a tuple for the unanswered queries which see CERN as a viable option per week.

PCA_DUMP_FOLDER
	This is a folder path where various pipeline results are dumped for the PCA method.

RAW_QUERY_FILES
	This is a tuple containing the path to the CSV files containing the raw query files per week.
'''

MATRIXES_FOLDER_LIST = (\
	'./matrices_folder/remote_host_0/log_folder/',\
	'./matrices_folder/remote_host_1/log_folder/',\
	'./matrices_folder/remote_host_1/log_folder/',\
	'./log_folder_13th_may/'
)

# WEEK_TIME_MOMENTS = (\
# 	(1579215600000, 1579875041000),\
# 	(1580511600000, 1581289199000),\
# 	(1581289200000, 1581980399000),\
# )

# vvvv     Third week is shortened     vvvv
# vvvv                                 vvvv
WEEK_TIME_MOMENTS = (\
	(1579215600000, 1579875041000),\
	(1580511600000, 1581289199000),\
	(1581289200000, 1581670035995),\
	(1589398710001, 1590609266276),\
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

CSV_THROUGHPUT_FILES = (\
	'./throughput_folder/january_month_throughput.csv',\
	'./throughput_folder/february_month.csv',\
	'./throughput_folder/february_month.csv',\
	'./throughput_folder/throughput_may.csv',\
)

BINARY_TREND_FILES = (\
	'./trend_folder/week_0.p',\
	'./trend_folder/week_1.p',\
	'./trend_folder/week_2.p',\
	'./trend_folder/week_3.p',\
)

def dump_new_set(dump_data_set_name, read_size_per_tm_file):
	'''
	This associates trend, average read size, demotion and distance matrices based on time tags.
	The throughput files are not calculated from MonALISA, but instead they are downloaded from
	the alimonitor site.

	dump_data_set_name
		The name of the file where the results are dumped

	read_size_per_tm_file
		The name of the file which contains the the read size per time moment from the queries.
	'''

	# Set up throughput
	if False:
		thp_iterable_tuple = (
			get_thp('./throughput_folder/january_month_throughput.csv',WEEK_TIME_MOMENTS[0]),
			get_thp('./throughput_folder/february_month.csv',WEEK_TIME_MOMENTS[1]),
			get_thp('./throughput_folder/february_month.csv',WEEK_TIME_MOMENTS[2])
		)
	if True:
		thp_iterable_tuple =\
			tuple(
				map(
					lambda file_path: pickle.load( open( file_path , 'rb' ) ),
					BINARY_TREND_FILES
				)
			)
	print('Finished throughput set up !')

	# Set up all read_size_per_time_moment
	if True:
		read_size_iterable_tuple = sorted(pickle.load( open( PCA_DUMP_FOLDER + read_size_per_tm_file, 'rb' )))
	if False:
		read_size_iterable_tuple = [ None for _ in WEEK_TIME_MOMENTS ]
	print('Finished read size per time moment set up !')

	get_matrix_f =\
	lambda folder_name, keyword, time_moments, extraction_f:\
		tuple(\
			map(\
				lambda time_tag: ( time_tag , extraction_f( time_tag , folder_name )  ),\
				sorted(\
					filter(\
						lambda t: time_moments[0] - 25200000 <= t < time_moments[1] + 25200000,\
						map(\
							lambda fn: int( fn.split('_')[0] ),\
							filter(\
								lambda fn: keyword in fn,\
								os.listdir(folder_name)\
							)\
						)\
					)\
				)\
			)\
		)

	# Set up distance matrices
	distance_iterable_tuple = list()
	for ind in range(len(MATRIXES_FOLDER_LIST)):
		distance_iterable_tuple.append(
			get_matrix_f(
				MATRIXES_FOLDER_LIST[ind],\
				'distance',\
				WEEK_TIME_MOMENTS[ind],\
				get_dist_dict_by_time,\
			)
		)
		print('Finished for', ind + 1, '/', len(MATRIXES_FOLDER_LIST))

	print('Finished distance matrices set up !')

	# Set up demotion matrices
	demotion_iterable_tuple = list()
	for ind in range(len(MATRIXES_FOLDER_LIST)):
		demotion_iterable_tuple.append(
			get_matrix_f(\
				MATRIXES_FOLDER_LIST[ind],\
				'demotion',\
				WEEK_TIME_MOMENTS[ind],\
				get_dem_dict_by_time
			)
		)
		print('Finished for', ind + 1, '/', len(MATRIXES_FOLDER_LIST))
	print('Finished distance matrices set up !')

	data_set_list = list()

	for i in range(len(MATRIXES_FOLDER_LIST)):
		data_set_list.append(
			create_data_set(
				thp_iterable_tuple[i],
				read_size_iterable_tuple[i],
				distance_iterable_tuple[i],
				demotion_iterable_tuple[i],
				ignore_read_size_flag=False,
			)
		)

		print( len( data_set_list[-1] ) )

	print( 'Total is '\
		+ str( sum( map( lambda e: len(e) , data_set_list ) ) ) )

	pickle.dump(
		data_set_list,
		open( PCA_DUMP_FOLDER + dump_data_set_name , 'wb' )
	)

def get_whole_matrixes(input_file_name, output_file_name):
	'''
	Adds the distance and demotion matrices together.
	Creates a minimal set of storage elements and a minimal set of clients that are present
		in all the distance and demotion matrices (the matrices are stored as Python dictionaries).
	'''
	data_set_list = list()

	for week_list in pickle.load(open(PCA_DUMP_FOLDER + input_file_name,'rb')):

		new_week_list = list()

		for tm, thp_v, ars, dist_d, dem_d in week_list:

			for client_name in dist_d.keys():

				for se_name in dist_d[client_name].keys():

					dist_d[client_name][se_name] += dem_d[se_name]

			new_week_list.append(
				(
					tm,
					thp_v,
					ars,
					dist_d,
				)
			)

		data_set_list.append( new_week_list )

	cl_list, se_list = get_clients_and_ses_minimal_list_1(
		tuple(
			map(
				lambda week_list:\
					list(map(lambda e: e[-1], week_list)),\
				data_set_list
			)
		)
	)

	new_data_set = list()

	for week_list in data_set_list:
		new_week_list = list()
		for tm, thp_v, ars, dist_d in week_list:
			whole_matrix_list = list()

			for cl_name in cl_list:
				for se_name in se_list:
					whole_matrix_list.append(
						dist_d[cl_name][se_name]
					)

			new_week_list.append(
				(
					tm,
					thp_v,
					ars,
					whole_matrix_list,
				)
			)

		new_data_set.append( new_week_list )

	pickle.dump(
		new_data_set,
		open( PCA_DUMP_FOLDER + output_file_name , 'wb' )
	)

def get_principal_components(input_file_name, output_file_name):
	'''
	This dumps some statistics on the difference in elements and the principal components.
	'''

	# tm_thp_ars_whole_per_week_4_may - reduced from third week matrices

	data_set_list = pickle.load(open(PCA_DUMP_FOLDER + input_file_name,'rb'))

	# print( len( data_set_list[0][0][-1] ) )

	if False:
		whole_matrix_array = np.array(
			list(map(lambda e: e[-1], data_set_list[0]))\
			+ list(map(lambda e: e[-1], data_set_list[1]))\
			+ list(map(lambda e: e[-1], data_set_list[2]))\
		)

	if True:
		whole_matrix_array = list()

		for week_list in data_set_list:

			for e in week_list:

				is_in_whole_matrix_array_flag = False

				for el in whole_matrix_array:

					is_same_flag = True

					for p in zip( e[-1] , el ):
						if p[1] != p[0]:
							is_same_flag = False
							break

					if is_same_flag:
						is_in_whole_matrix_array_flag = True
						break

				if not is_in_whole_matrix_array_flag:
					whole_matrix_array.append( e[-1] )

		print('Whole matrix length: ' + str(len(whole_matrix_array)))

		whole_matrix_array = np.array( whole_matrix_array )

	def dump_cell_diff( cell_list , name='pipe_whole_cell_diff_4_may.p' ):
		cell_diff_list = []

		for previous_v, next_v in zip( cell_list[:-1] , cell_list[1:] ):

			v = 0

			for v1, v2 in zip( previous_v , next_v ):

				if v1 != v2:

					v += 1

			cell_diff_list.append( v )

		pickle.dump(
			cell_diff_list,
			open(
				name,
				'wb'
			)
		)

	if False:
		dump_cell_diff(whole_matrix_array)

	pca_engine = PCA(11)

	pca_engine.fit(
		whole_matrix_array
	)

	print(pca_engine.explained_variance_ratio_.cumsum())

	exit(0)

	if False:
		dump_cell_diff(
			pca_engine.transform( whole_matrix_array )[:,:11],
			'pipe_pc_cell_diff_4_may.p'
		)

	new_data_set = list()
	for week_list in data_set_list:

		new_week_list = list()

		for p in zip( week_list , pca_engine.transform( np.array( list(map(lambda e: e[-1], week_list) ) ) ) ):

			new_week_list.append(
				(
					p[0][0],
					p[0][1],
					p[0][2],
					list( p[1] ),
				)
			)

		new_data_set.append( new_week_list )

	# 'tm_tren_ars_pc_per_week_10_may.p'
	pickle.dump(
		new_data_set,
		open(
			PCA_DUMP_FOLDER + output_file_name,
			'wb'
		)
	)

	# 4 mai - 6.p - 14000 de exemple, pca cu toate matricele la examinare, reducere cu cele 2 zile
	# 10 mai - 8.p - 14000 de exemple, pca numai cu unice, reducere cu cele 2 zile

def generate_data_set(input_file, output_file):
	'''
	Does normalization and dumps
	'''

	indsexes, a = normalize_and_split_data_set_1(
		input_file,
		40,
		39,
		True
	)
	a,b,c,d = a


	pickle.dump(
		{
			'train_valid_data_sets':\
				(
					a,
					b,
					c,
					d,
				),
			'train_valid_indexes':\
				indsexes,\
		},
		open(
			'./pca_data_sets/' + output_file,
			'wb'
		)
	)

def analyse_main_0():
	'''
	Looks into demotion and distance matrix files and prints
	time tags separated by at leas one minute.
	'''
	get_time_tags_f = lambda time_moments , keyword , folder_name : sorted(\
		filter(\
			lambda t: time_moments[0] <= t < time_moments[1],\
			map(\
				lambda fn: int( fn.split('_')[0] ),\
				filter(\
					lambda fn: keyword in fn,\
					os.listdir(folder_name)\
				)\
			)\
		)\
	)\

	a_list =\
	get_time_tags_f(\
		WEEK_TIME_MOMENTS[2],
		'distance',
		MATRIXES_FOLDER_LIST[2],
	)

	for ind, tm_0, tm_1 in zip( range( 1 , len(a_list) ) , a_list[:-1] , a_list[1:] ):
		if tm_1 - tm_0 > 1000 * 60000:
			print( str(ind) + ' ' + str(tm_0) + ' ' + str(tm_1) )

	# 6353 1581670035995 1581891557935

def dump_set_for_encoder_decoder(input_file_name, window_size, output_file_name, ignore_read_size_flag=False):
	'''
	Creates the final data set for training and validation. It writes to disk a pickle file which
	contains the set of "whole" (the addition of the distance and demotion matrices) matrices
	ordered in time and the train/validation indeces. The reason why the data set is not written
	to disk in a time sequence manner is because it would take too much space. (~ 6700 x 40 x 14000).
	So instead, a generator is used for training.

	input_file_name
		The filename where the whole matrices are stored

	window_size
		The time sequence length (40)

	output_file_name
		The data set dictionary

	ignore_read_size_flag
		Boolean on whether to include or not the average read size over 2 minutes
	'''
	whole_matrices_data_set = pickle.load(open(PCA_DUMP_FOLDER+input_file_name,'rb'))

	min_matrix = max_matrix = whole_matrices_data_set[0][0][-1][0]

	min_ars = max_ars = whole_matrices_data_set[0][0][2]

	min_t = max_t = whole_matrices_data_set[0][0][1]

	for week_list in whole_matrices_data_set:

		for _, tre, ars, whole_m in week_list:

			if tre < min_t: min_t = tre
			if tre > max_t: max_t = tre

			if not ignore_read_size_flag:
				if ars < min_ars: min_ars = ars
				if ars > max_ars: max_ars = ars

			if min(whole_m) < min_matrix: min_matrix = min(whole_m)
			if max(whole_m) > max_matrix: max_matrix = max(whole_m)

	print('min and max are', min_ars, max_ars)

	new_data_set = list()

	train_indexes_list = list()
	valid_indexes_list = list()

	last_limit = 0

	for week_list in whole_matrices_data_set:

		valid_indexes_list +=\
			random.sample(
				range( last_limit + window_size, last_limit + len(week_list) ),
				round( 0.2 * ( len(week_list) - 40 ) ),
			)

		train_indexes_list +=\
			list(
				filter(
					lambda i: i not in valid_indexes_list,
					range( last_limit + window_size, last_limit + len(week_list) )
				)
			)

		last_limit += len(week_list)

		for _, tre, ars, whole_m in week_list:

			if ignore_read_size_flag:
				new_data_set.append(list())
			else:
				new_data_set.append(
					[ 2 * ( ars - min_ars ) / ( max_ars - min_ars ) - 1, ]
				)


			for v in whole_m:

				new_data_set[-1].append(
					2 * ( v - min_matrix ) / ( max_matrix - min_matrix ) - 1
				)

			new_data_set[-1].append(
				( tre - min_t ) / ( max_t - min_t )
			)

	del whole_matrices_data_set

	print( 'Number of elements per example: ' + str( len( new_data_set[0] ) ) )

	pickle.dump(
		{
			'train_indexes' : train_indexes_list,
			'valid_indexes' : valid_indexes_list,
			'non_split_data_set' : new_data_set,
		},
		open(
			'./pca_data_sets/' + output_file_name , 'wb'
		)
	)

def generate_random_batch(batch_size, window_size, indexes_list):
	'''
	Generates a batch from the global "data_set_dict".
	'''
	x_list = list()
	y1_list = list()
	y2_list = list()

	for index in random.sample( indexes_list , batch_size ):

		x_list.append( list() )
		y1_list.append( list() )
		y2_list.append( list() )

		for a_list in data_set_dict['non_split_data_set'][index-window_size:index]:

			x_list[-1].append( a_list[:-1] )
			y1_list[-1].append( [ a_list[-1] , ] )
			y2_list[-1].append( a_list[:-1] )

	return\
		np.array(x_list),\
		[\
			np.array(y1_list),\
			np.array(y2_list),\
		]

def gen_train(batch_size, window_size):
	while True:
		yield generate_random_batch_1(\
			batch_size,
			window_size,
			data_set_dict['train_indexes'],
			only_last_flag=False,
			one_input_flag=False
		)

def gen_valid(batch_size, window_size):
	while True:
		yield generate_random_batch_1(\
			batch_size,
			window_size,
			data_set_dict['valid_indexes'],
			only_last_flag=False,
			one_input_flag=False
		)

def train_for_encoder_decoder():
	'''
	Training of a single configuration for the encoder-decoder neural network
	'''

	# with tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"]).scope():
	if True:

		global data_set_dict
		data_set_dict = pickle.load(open(
			'pca_data_sets/8_28_may.p',
			'rb'
		))

		data_set_dict['non_split_data_set'] = np.array( data_set_dict['non_split_data_set'] )

		ws = 40

		if False:
			bins_no = 6769
			inp_layer = keras.layers.Input(shape=(ws, bins_no,))
			x = keras.layers.TimeDistributed(
				keras.layers.Dense(
					units=10,
					activation='relu'
				)
			)(inp_layer)
			last_common_x = keras.layers.BatchNormalization()(x)

			y2 = keras.layers.TimeDistributed(
				keras.layers.Dense(
					units=bins_no,
					activation='tanh'
				)
			)(last_common_x)

			x = keras.layers.Dropout(0.1)(last_common_x)

			x = keras.layers.Bidirectional(
				keras.layers.LSTM(
					units=1,
					return_sequences=False,
				)
			)(x)
			x = keras.layers.BatchNormalization()(x)

			# y1 = keras.layers.TimeDistributed(
			# 	keras.layers.Dense(
			# 		units=1,
			# 		activation='sigmoid'
			# 	)
			# )(x)

			y2 = keras.layers.Dense(
				units=1,
				activation='sigmoid',
			)(x)

			model = keras.models.Model(inputs=inp_layer, outputs=[y1,y2,])

		if True:
			bins_no = 6768

			inp_layer_1 = keras.layers.Input(shape=(ws, 1,))

			inp_layer_2 = keras.layers.Input(shape=(ws, bins_no,))

			x = inp_layer_2

			# x = keras.layers.TimeDistributed(
			# 	keras.layers.Dense(
			# 		units=40,
			# 		activation='relu'
			# 	)
			# )(inp_layer_2)
			# x = keras.layers.BatchNormalization()(x)
			# x = keras.layers.Dropout(0.4)(x)

			# x = keras.layers.TimeDistributed(
			# 	keras.layers.Dense(
			# 		units=20,
			# 		activation='relu'
			# 	)
			# )(x)
			# x = keras.layers.BatchNormalization()(x)
			# x = keras.layers.Dropout(0.4)(x)

			x = keras.layers.TimeDistributed(
				keras.layers.Dense(
					units=10,
					activation='relu'
				)
			)(x)
			last_common_x = keras.layers.BatchNormalization()(x)

			x = last_common_x

			# x = keras.layers.TimeDistributed(
			# 	keras.layers.Dense(
			# 		units=20,
			# 		activation='relu'
			# 	)
			# )(last_common_x)
			# x = keras.layers.BatchNormalization()(x)

			# x = keras.layers.TimeDistributed(
			# 	keras.layers.Dense(
			# 		units=40,
			# 		activation='relu'
			# 	)
			# )(x)
			# x = keras.layers.BatchNormalization()(x)

			y1 = keras.layers.TimeDistributed(
				keras.layers.Dense(
					units=bins_no,
					activation='tanh'
				)
			)(x)

			x = keras.layers.Concatenate()([ inp_layer_1 , last_common_x ])
			x = keras.layers.Dropout(0.4)(x)

			# x = keras.layers.TimeDistributed(
			# 	keras.layers.Dense(
			# 		units=5,
			# 		activation='relu'
			# 	)
			# )(x)
			# x = keras.layers.BatchNormalization()(x)
			# x = keras.layers.Dropout(0.4)(x)

			x = keras.layers.Bidirectional(
				keras.layers.LSTM(
					units=1,
					return_sequences=True,
					# kernel_regularizer=keras.regularizers.l2(0.001),
					# bias_regularizer=keras.regularizers.l2(0.001),
					# activity_regularizer=keras.regularizers.l2(0.00001)
				)
			)(x)
			x = keras.layers.BatchNormalization()(x)

			# y2 = keras.layers.Dense(
			# 	units=1,
			# 	activation='sigmoid',
			# )(x)

			y2 = keras.layers.TimeDistributed(
				keras.layers.Dense(
					units=1,
					activation='sigmoid'
				)
			)(x)

			model = keras.models.Model(inputs=[inp_layer_1,inp_layer_2,], outputs=[y1,y2,])

		model.summary()

		model.compile(
			optimizer=keras.optimizers.Adam(),
			loss='mean_absolute_percentage_error',
			metrics=['mae',]
		)

		if False:
			csv_log_path = './pca_csv_folder/losses_encoder_decoder.csv'
			model_path_save = './pca_multiple_model_folders/encoder_decoder_models/'

		if True:
			csv_log_path = './pca_csv_folder/losses_encoder_decoder_1.csv'
			model_path_save = './pca_multiple_model_folders/encoder_decoder_models_1/'

		if True:
			model.fit_generator(
				gen_train(128, ws),
				epochs=2000,
				steps_per_epoch=100,
				validation_data=gen_valid(128,ws),
				validation_steps=20,
				verbose=2,
				callbacks=[
					keras.callbacks.CSVLogger( csv_log_path ),
					keras.callbacks.ModelCheckpoint(
						model_path_save + "model_{epoch:04d}.hdf5",
						monitor='val_loss',
						save_best_only=True
					)
				]
			)

		if False:
			print( 'Will start to build data set !' )

			x_train, y_train =\
				generate_random_batch_1(
					len(data_set_dict['train_indexes']),
					# 16,
					ws,
					data_set_dict['train_indexes'],
					only_last_flag=True,
					firs_bs_elements_flag=True,
					log_freq=1000,
				)
			x_valid, y_valid =\
				generate_random_batch_1(
					len(data_set_dict['valid_indexes']),
					# 16,
					ws,
					data_set_dict['valid_indexes'],
					only_last_flag=True,
					firs_bs_elements_flag=True,
					log_freq=1000,
				)

			print( 'Will start trainin\' !' )

			model.fit(
				x=x_train,
				y=y_train,
				batch_size=2,
				epochs=2000,
				validation_data=(\
					x_valid,
					y_valid,
				),
				verbose=2,
				callbacks=[
					keras.callbacks.CSVLogger( csv_log_path ),
					keras.callbacks.ModelCheckpoint(
						model_path_save + "model_{epoch:04d}.hdf5",
						monitor='val_loss',
						save_best_only=True
					)
				]
			)

def dump_results_for_plot():
	'''
	Writes to disk the loss evolution of a model during training.
	'''
	history_gen = csv.reader( open( './pca_csv_folder/losses_328.csv' , 'rt' ) )

	next(history_gen)

	train_mape_list = list()
	valid_mape_list = list()
	train_mae_list = list()
	valid_mae_list = list()

	for line_list in history_gen:
		if len(line_list) > 2:
			train_mape_list.append( float( line_list[4] ) )
			valid_mape_list.append( float( line_list[9] ) )
			train_mae_list.append( float( line_list[5] ) )
			valid_mae_list.append( float( line_list[10] ) )

	pickle.dump(
		{
			'validation_predictions' :\
				(
					ground_truth_list,
					pred_list

				),
			'mape_evolution' :\
				(
					train_mape_list,
					valid_mape_list,
				),
			'mae_evolution' :\
				(
					train_mae_list,
					valid_mae_list,
				),
		},
		open(
			'./pca_results/result_encoder_decoder_0.p', 'wb'
		)
	)

def dump_train_set_distribution():
	'''
	Creates the value distribution of the trend into different brackets.
	'''
	data_set_dict = pickle.load(open(
		'pca_data_sets/7.p',
		'rb'
	))

	def get_distr_dict(indexes_list):

		train_distribution_dict = {\
			( 0 , 0.05) : 0,
			( 0.05 , 0.1 ) : 0,
			( 0.1 , 0.15 ) : 0,
			( 0.15 , 0.2 ) : 0,
			( 0.2 , 0.25 ) : 0,
			( 0.25 , 0.3 ) : 0,
			( 0.3 , 0.35 ) : 0,
			( 0.35 , 0.4 ) : 0,
			( 0.4 , 0.45 ) : 0,
			( 0.45 , 0.5 ) : 0,
			( 0.5 , 0.55 ) : 0,
			( 0.55 , 0.6 ) : 0,
			( 0.6 , 0.65 ) : 0,
			( 0.65 , 0.7 ) : 0,
			( 0.7 , 0.75 ) : 0,
			( 0.75 , 0.8 ) : 0,
			( 0.8 , 0.85 ) : 0,
			( 0.85 , 0.9 ) : 0,
			( 0.9 , 0.95 ) : 0,
			( 0.95 , 1.05 ) : 0,
		}

		for index in indexes_list:
			for k in train_distribution_dict.keys():
				if k[0] <= data_set_dict['non_split_data_set'][index-1][-1] < k[1]:
					train_distribution_dict[k] += 1

		for k in train_distribution_dict.keys():
			train_distribution_dict[k] =\
				100 * train_distribution_dict[k] / len(indexes_list)

		return train_distribution_dict

	tr_d, va_d =\
		get_distr_dict(data_set_dict['train_indexes']),\
		get_distr_dict(data_set_dict['valid_indexes'])

	for k in tr_d.keys():
		print( k , tr_d[k] , va_d[k] , )

def get_time_differences_between_throughputs_in_minutes():
	'''
	Iterates through the throughput files and prints
	the diferences in logging time.
	'''
	differences_set = set()

	for csv_file in CSV_THROUGHPUT_FILES:
		thp_gen = csv.reader( open(csv_file, 'rt') )

		next(thp_gen)

		thp_line = next(thp_gen)
		while len(thp_line[-1]) < 2:
			thp_line = next(thp_gen)

		prev_time_stamp = int( thp_line[0] )

		for thp_line in thp_gen:

			if len(thp_line[-1]) > 2:

				new_ts = int( thp_line[0] )

				if new_ts - prev_time_stamp > 120:
					differences_set.add(
						( new_ts - prev_time_stamp ) / 60
					)

				prev_time_stamp = new_ts

	differences_set = sorted( differences_set )

	print(differences_set)

def generate_trend_files():
	'''
	Generates trend files.
	'''
	for thp_fn, ti, tr_fn in zip(CSV_THROUGHPUT_FILES, WEEK_TIME_MOMENTS, BINARY_TREND_FILES):
		pickle.dump(
			get_thp_1(thp_fn, ti),
			open(
				tr_fn, 'wb'
			)
		)

def plot_thp_trend_for_final_report():
	'''
	Used to showcase a comparison between the original throughput and the trend.
	'''
	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)
	from statsmodels.tsa.seasonal import seasonal_decompose

	def get_raw_throughput(thp_file, time_interval):
		print('Entered',thp_file)

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

		# print(len(thp_list))

		new_thp_list.append( thp_list[-1] )

		return new_thp_list

	throughput_iterable_tuple = (
		get_raw_throughput('./throughput_folder/january_month_throughput.csv',WEEK_TIME_MOMENTS[0]),
		get_raw_throughput('./throughput_folder/february_month.csv',WEEK_TIME_MOMENTS[1]),
		get_raw_throughput('./throughput_folder/february_month.csv',WEEK_TIME_MOMENTS[2])
	)

	def get_trend(thp_list,time_interval):
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

	def normalize_thingy(thingy):
		min_v = min(map(lambda e: min(e,key=lambda f: f[1])[1], thingy))
		max_v = max(map(lambda e: max(e,key=lambda f: f[1])[1], thingy))

		return\
			tuple(
				map(
					lambda e:\
						tuple(
							map(
								lambda p: ( p[1] - min_v ) / ( max_v - min_v ),
								e
							)
						),
					thingy
				)
			)

	trend_iterable_tuple = tuple(\
		map(\
			lambda e: get_trend(e[0],e[1]),\
			zip(throughput_iterable_tuple,WEEK_TIME_MOMENTS[:3])
			)
		)

	throughput_iterable_tuple = tuple(\
		map(\
			lambda e: list(filter(lambda f: e[1][0] <= f[0] < e[1][1], e[0])),\
			zip(throughput_iterable_tuple,WEEK_TIME_MOMENTS[:3])
			)
		)


	if True:
		trend_iterable_tuple = normalize_thingy(trend_iterable_tuple)
		throughput_iterable_tuple = normalize_thingy(throughput_iterable_tuple)
		get_thp_from_el = lambda e: e
	else:
		get_thp_from_el = lambda e: e[1]

	plt.plot(
		range(len(throughput_iterable_tuple[0])),
		tuple(map(lambda e: get_thp_from_el(e), throughput_iterable_tuple[0])),
		'b-',
		label='Normalized Throughput'
	)
	plt.plot(
		range(len(trend_iterable_tuple[0])),
		tuple(map(lambda e: get_thp_from_el(e), trend_iterable_tuple[0])),
		'g-',
		label='Normalized Trend'
	)
	offset = len(throughput_iterable_tuple[0])
	plt.plot(\
		range(offset, offset + len(throughput_iterable_tuple[1])),\
		throughput_iterable_tuple[1],
		'b-'
	)
	plt.plot(\
		range(offset, offset + len(trend_iterable_tuple[1])),\
		trend_iterable_tuple[1],
		'g-'
	)
	offset += len(throughput_iterable_tuple[1])
	plt.plot(\
		range(offset, offset + len(throughput_iterable_tuple[2])),\
		throughput_iterable_tuple[2],
		'b-'
	)
	plt.plot(\
		range(offset, offset + len(trend_iterable_tuple[2])),\
		trend_iterable_tuple[2],
		'g-'
	)
	plt.plot(
		( 0 , 0 , ),
		( 0 , 1 , ),
		'r-',
		label='Week Limiter'
	)
	plt.plot(
		( 0 , len(trend_iterable_tuple[0]) ),
		( 0 , 0),
		'r-'
	)
	plt.plot(
		( len(trend_iterable_tuple[0]) , len(trend_iterable_tuple[0]) ),
		( 0 , 1),
		'r-'
	)
	plt.plot(
		( 0 , len(trend_iterable_tuple[0]) ),
		( 1 , 1),
		'r-'
	)
	plt.plot(
		( len(trend_iterable_tuple[0]) ,\
			len(trend_iterable_tuple[0]) + len(trend_iterable_tuple[1]) ),
		( 0 , 0),
		'r-'
	)
	plt.plot(
		( len(trend_iterable_tuple[0]) ,\
			len(trend_iterable_tuple[0]) + len(trend_iterable_tuple[1]) ),
		( 1 , 1),
		'r-'
	)
	plt.plot(
		(\
			len(trend_iterable_tuple[0]) + len(trend_iterable_tuple[1]),\
			len(trend_iterable_tuple[0]) + len(trend_iterable_tuple[1]),\
		),
		( 0 , 1),
		'r-'
	)
	plt.plot(
		(\
			len(trend_iterable_tuple[0]) + len(trend_iterable_tuple[1]),\
			len(trend_iterable_tuple[0]) + len(trend_iterable_tuple[1])\
				+ len(trend_iterable_tuple[2]),\
		),
		( 0 , 0),
		'r-'
	)
	plt.plot(
		(\
			len(trend_iterable_tuple[0]) + len(trend_iterable_tuple[1]),\
			len(trend_iterable_tuple[0]) + len(trend_iterable_tuple[1])\
				+ len(trend_iterable_tuple[2]),\
		),
		( 1 , 1),
		'r-'
	)
	plt.plot(
		(\
			len(trend_iterable_tuple[0]) + len(trend_iterable_tuple[1])\
				+ len(trend_iterable_tuple[2]),\
			len(trend_iterable_tuple[0]) + len(trend_iterable_tuple[1])\
				+ len(trend_iterable_tuple[2]),\
		),
		( 0 , 1),
		'r-'
	)
	plt.legend()
	plt.xlabel('Index')
	plt.ylabel(' Normalized Value')
	plt.show()

if __name__ == '__main__':

	# dump_new_set(\
	# 	'tm_trend_ars_dist_dem_4th_june.p',\
	# 	'all_queries_read_size_4th_june.p'
	# )
	# analyse_main_0()
	# get_whole_matrixes(\
	# 	'tm_trend_ars_dist_dem_4th_june.p',\
	# 	'tm_trend_ars_whole_per_week_4th_june.p'\
	# )
	# get_principal_components(
	# 	'tm_trend_ars_whole_per_week_4th_june.p',
	# 	'tm_tren_ars_pc_per_week_4th_june.p'
	# )
	# get_principal_components(
	# 	'tm_thp_ars_whole_per_week_4_may.p',
	# 	'tm_tren_ars_pc_per_week_4th_june.p'
	# )
	# generate_data_set(
	# 	'tm_tren_ars_pc_per_week_4th_june.p',
	# 	'9_4th_June_pc.p'
	# )

	# dump_set_for_encoder_decoder(
	# 	'tm_trend_ars_whole_per_week_4th_june.p',
	# 	40,
	# 	'8_4th_june.p',
	# )
	# train_for_encoder_decoder()

	# dump_results_for_plot()

	# dump_train_set_distribution()

	# generate_trend_files()

	plot_thp_trend_for_final_report()