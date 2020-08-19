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
import sys

# Skylate paths
MATRIXES_FOLDER_LIST = (\
	'./remote_host_0/log_folder/',\
	'./remote_host_1/log_folder/',\
	'./remote_host_1/log_folder/',\
)

WEEK_TIME_MOMENTS = (\
	(1579215600000, 1579875041000),\
	(1580511600000, 1581289199000),\
	(1581289200000, 1581980399000),\
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

def get_small_encoder_decoder_model_0(arg_dict):
	'''
	Creates and returns a model for the encoder-decoder architecture.

	arg_dict
		Dictionary that contains various tags that alter the
		neural network structure.
	'''
	print( '\n\n\n\nWill construct model for', arg_dict['new_index'] , '!\n\n\n\n' )

	bins_no = 6769

	inp_layer_2 = keras.layers.Input(shape=(40, bins_no,))

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=arg_dict['latent_values_count'],
			activation='relu'
		)
	)(inp_layer_2)
	last_common_x = keras.layers.BatchNormalization()(x)

	y1 = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=bins_no,
			activation='tanh'
		)
	)(last_common_x)

	x = keras.layers.Dropout(arg_dict['dropout_value'])(last_common_x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=1,
			return_sequences=True,
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)

	y2 = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='sigmoid'
		)
	)(x)

	model = keras.models.Model(inputs=inp_layer_2, outputs=[y1,y2,])

	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	print( '\n\n\n\nWill exit model creation for', arg_dict['new_index'] , '!\n\n\n\n' )

	return model

def get_small_encoder_decoder_model_1(arg_dict):
	'''
	Creates and returns a model for the encoder-decoder architecture.

	arg_dict
		Dictionary that contains various tags that alter the
		neural network structure.
	'''
	print( '\n\n\n\nWill construct model for', arg_dict['new_index'] , '!\n\n\n\n' )

	inp_layer_1 = keras.layers.Input(shape=(40, 1,))
	inp_layer_2 = keras.layers.Input(shape=(40, arg_dict['bins_no'],))

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=arg_dict['latent_v_count'],
			activation='relu'
		)
	)(inp_layer_2)
	last_common_x = keras.layers.BatchNormalization()(x)

	y1 = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=arg_dict['bins_no'],
			activation='tanh'
		)
	)(last_common_x)

	x = keras.layers.Concatenate()([ inp_layer_1 , last_common_x ])

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=arg_dict['latent_v_count'],
			activation='relu')
	)(x)
	x = keras.layers.BatchNormalization()(x)
	# x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=arg_dict['latent_v_count'],
			return_sequences=True,
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	# x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=arg_dict['latent_v_count'],
			activation='relu')
	)(x)
	x = keras.layers.BatchNormalization()(x)
	# x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=5,
				activation='relu')
	)(x)
	x = keras.layers.BatchNormalization()(x)
	# x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=1,
			return_sequences=False,
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)

	y2 = keras.layers.Dense(
		units=1,
		activation='sigmoid'
	)(x)

	model = keras.models.Model(inputs=[inp_layer_1, inp_layer_2], outputs=[y1,y2,])

	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	print( '\n\n\n\nWill exit model creation for', arg_dict['new_index'] , '!\n\n\n\n' )

	return model

def get_small_encoder_decoder_model_1_with_l2_regularizer(arg_dict):
	'''
	Creates and returns a model for the encoder-decoder architecture.

	arg_dict
		Dictionary that contains various tags that alter the
		neural network structure.
	'''
	print( '\n\n\n\nWill construct model for', arg_dict['new_index'] , '!\n\n\n\n' )

	inp_layer_1 = keras.layers.Input(shape=(40, 1,))
	inp_layer_2 = keras.layers.Input(shape=(40, arg_dict['bins_no'],))

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=arg_dict['latent_v_count'],
			activation='relu',
		)
	)(inp_layer_2)
	last_common_x = keras.layers.BatchNormalization()(x)

	y1 = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=arg_dict['bins_no'],
			activation='tanh'
		)
	)(last_common_x)

	x = keras.layers.Concatenate()([ inp_layer_1 , last_common_x ])

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=10,
			activation='relu',
		    kernel_regularizer=keras.regularizers.l2(1e-4),
			bias_regularizer=keras.regularizers.l2(1e-4),
			activity_regularizer=keras.regularizers.l2(1e-5)
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	# x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=10,
			return_sequences=True,
		    kernel_regularizer=keras.regularizers.l2(1e-4),
			bias_regularizer=keras.regularizers.l2(1e-4),
			activity_regularizer=keras.regularizers.l2(1e-5)
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	# x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=10,
			activation='relu',
		    kernel_regularizer=keras.regularizers.l2(1e-4),
			bias_regularizer=keras.regularizers.l2(1e-4),
			activity_regularizer=keras.regularizers.l2(1e-5)
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	# x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=5,
			activation='relu',
		    kernel_regularizer=keras.regularizers.l2(1e-4),
			bias_regularizer=keras.regularizers.l2(1e-4),
			activity_regularizer=keras.regularizers.l2(1e-5)
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	# x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=1,
			return_sequences=False,
		    kernel_regularizer=keras.regularizers.l2(1e-4),
			bias_regularizer=keras.regularizers.l2(1e-4),
			activity_regularizer=keras.regularizers.l2(1e-5)
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)

	y2 = keras.layers.Dense(
		units=1,
		activation='sigmoid',
	    kernel_regularizer=keras.regularizers.l2(1e-4),
		bias_regularizer=keras.regularizers.l2(1e-4),
		activity_regularizer=keras.regularizers.l2(1e-5)
	)(x)

	model = keras.models.Model(inputs=[inp_layer_1, inp_layer_2], outputs=[y1,y2,])

	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	print( '\n\n\n\nWill exit model creation for', arg_dict['new_index'] , '!\n\n\n\n' )

	return model

def load_last_model(arg_dict):
	'''
	During the training process, models are saved to disk. This function loads the last
	model. The "new_index" tag identifies the series of models.

	arg_dict
		Dictionary
	'''
	models_paths_list = os.listdir( './pca_multiple_model_folders/models_' + str(arg_dict['new_index']) )
	if len(models_paths_list) == 0:
		return get_small_encoder_decoder_model_1(arg_dict)
	return keras.models.load_model(
		'./pca_multiple_model_folders/models_' + str(arg_dict['new_index']) + '/'\
		+ max(map(lambda fn: ( int(fn[6:10]) , fn ) , models_paths_list ))[1]
	)

def encoder_decoder_fit_function(model, model_fit_dict):
	'''
	Used to fit a model. One can observe the "gen_train"
	and "gen_valid". These functions are used as generators,
	because loading the whole data set into memory takes up
	too much space.

	model
		Keras model to be fitted

	model_fit_dict
		Dictionary containing different options for the
		training process
	'''
	def gen_train(batch_size, window_size):
		while True:
			yield generate_random_batch_1(\
				batch_size,
				window_size,
				data_set_dict['enc_dec_set']['train_indexes'],
				data_set_dict['enc_dec_set'],
				only_last_flag=True,
				one_input_flag=False
			)

	def gen_valid(batch_size, window_size):
		while True:
			yield generate_random_batch_1(\
				batch_size,
				window_size,
				data_set_dict['enc_dec_set']['valid_indexes'],
				data_set_dict['enc_dec_set'],
				only_last_flag=True,
				one_input_flag=False,
			)

	print( '\n\n\n\nWill fit for', model_fit_dict['new_index'] , '!\n\n\n\n' )

	model.summary()

	model.fit_generator(
		gen_train(128, 40),
		epochs=model_fit_dict['epochs'],
		steps_per_epoch=100,
		validation_data=gen_valid(128,40),
		validation_steps=20,
		verbose=0,
		callbacks=[
			keras.callbacks.CSVLogger(
				model_fit_dict['csv_log_path']
			),
			keras.callbacks.ModelCheckpoint(
				model_fit_dict['models_dump_path'] + "model_{epoch:04d}.hdf5",
				monitor='val_loss',
				save_best_only=True
			),
			keras.callbacks.EarlyStopping(
				monitor='val_loss',
				patience=model_fit_dict['patience'],
			),
		]
	)

	print(  '\n\n\n\nFinished fit for', model_fit_dict['new_index'] , '!\n\n\n\n'  )

def get_encoder_decoder_parameters_0(new_index, get_log_str_function):
	'''
	Creates list of dictionaries containing different tag values (e.g. latent space dimension,
	dropout values).

	new_index
		hyperparameter configuration index

	get_log_str_function
		function translating the current configuration to a string in order to
		store the meaning of the configuration index
	'''
	pool_arguments_list = list()

	for lat_var_c in (10,15,20,):
		for dr_val in (0.1, 0.3, 0.4):

			with open( 'pca_index_meaning/' + str(new_index) + '.txt' , 'wt' ) as myfile:

				myfile.write(
					get_log_str_function(
						new_index,
						lat_var_c,
						dr_val,
					)
				)

			models_folder_name = 'models_' + str(new_index)

			if models_folder_name not in os.listdir('./pca_multiple_model_folders'):
				os.mkdir(
					'./pca_multiple_model_folders/'\
					+ models_folder_name
				)

			pool_arguments_list.append(
				(
					get_small_encoder_decoder_model_1,
					{
						'one_input_flag' : True,
						'latent_values_count' : lat_var_c,
						'only_last_flag' : False,
						'dropout_value' : dr_val,
						'new_index' : new_index,
					},
					encoder_decoder_fit_function,
					{
						'csv_log_path' :\
							'./pca_csv_folder/losses_'\
							+ str( new_index )
							+'.csv',
						'models_dump_path' :\
							'pca_multiple_model_folders/'\
							+ models_folder_name + '/',
						'epochs' : 300,
						'patience' : 200,
						'new_index' : new_index,
					},
				)
			)

			new_index += 1

	return pool_arguments_list, new_index

def get_encoder_decoder_parameters_1(new_index, get_log_str_function):
	'''
	Creates list of dictionaries containing different tag values (e.g. latent space dimension,
	dropout values).

	new_index
		hyperparameter configuration index

	get_log_str_function
		function translating the current configuration to a string in order to
		store the meaning of the configuration index
	'''
	pool_arguments_list = list()

	for dr_val in (0.1,):
		for lat_var_c in (9,10,11):
			for regularizer_flag in (True, False):

				if not ( dr_val == 0.4 and regularizer_flag == True ):

					with open( 'pca_index_meaning/' + str(new_index) + '.txt' , 'wt' ) as myfile:

						myfile.write(
							get_log_str_function(
								new_index,
								lat_var_c,
								dr_val,
								regularizer_flag
							)
						)

					models_folder_name = 'models_' + str(new_index)

					if models_folder_name not in os.listdir('./pca_multiple_model_folders'):
						os.mkdir(
							'./pca_multiple_model_folders/'\
							+ models_folder_name
						)

					if regularizer_flag:
						pool_arguments_list.append(
							(
								get_small_encoder_decoder_model_1_with_l2_regularizer,
								{
									'latent_v_count' : lat_var_c,
									'dropout_value' : dr_val,
									'new_index' : new_index,
									'bins_no' : 6072,
								},
								encoder_decoder_fit_function,
								{
									'csv_log_path' :\
										'./pca_csv_folder/losses_'\
										+ str( new_index )
										+'.csv',
									'models_dump_path' :\
										'pca_multiple_model_folders/'\
										+ models_folder_name + '/',
									'epochs' : 700,
									'patience' : 200,
									'new_index' : new_index,
								},
							)
						)
					else:
						pool_arguments_list.append(
							(
								load_last_model,
								{
									'latent_v_count' : lat_var_c,
									'dropout_value' : dr_val,
									'new_index' : new_index,
									'bins_no' : 6072,
								},
								encoder_decoder_fit_function,
								{
									'csv_log_path' :\
										'./pca_csv_folder/losses_'\
										+ str( new_index )
										+'.csv',
									'models_dump_path' :\
										'pca_multiple_model_folders/'\
										+ models_folder_name + '/',
									'epochs' : 700,
									'patience' : 200,
									'new_index' : new_index,
								},
							)
						)

					new_index += 1

	return pool_arguments_list, new_index

def get_encoder_decoder_parameters_2(new_index, get_log_str_function):
	'''
	Creates list of dictionaries containing different tag values (e.g. latent space dimension,
	dropout values).

	new_index
		hyperparameter configuration index

	get_log_str_function
		function translating the current configuration to a string in order to
		store the meaning of the configuration index
	'''
	pool_arguments_list = list()

	for dr_val in (0.1, 0.3, 0.4):
		for lat_var_c in (5,7,8):

			with open( 'pca_index_meaning/' + str(new_index) + '.txt' , 'wt' ) as myfile:

				myfile.write(
					get_log_str_function(
						new_index,
						lat_var_c,
						dr_val,
					)
				)

			models_folder_name = 'models_' + str(new_index)

			if models_folder_name not in os.listdir('./pca_multiple_model_folders'):
				os.mkdir(
					'./pca_multiple_model_folders/'\
					+ models_folder_name
				)

			pool_arguments_list.append(
				(
					get_small_encoder_decoder_model_1,
					{
						'latent_v_count' : lat_var_c,
						'dropout_value' : dr_val,
						'new_index' : new_index,
					},
					encoder_decoder_fit_function,
					{
						'csv_log_path' :\
							'./pca_csv_folder/losses_'\
							+ str( new_index )
							+'.csv',
						'models_dump_path' :\
							'pca_multiple_model_folders/'\
							+ models_folder_name + '/',
						'epochs' : 700,
						'patience' : 200,
						'new_index' : new_index,
					},
				)
			)

			new_index += 1

	return pool_arguments_list, new_index

def train_for_generator():
	'''

	DEPRECATED

	Was used to do a grid search in the hyperparameter space. This is part of a
	suboptimal approach. One can not train in parallel multiple models without
	creating new tensorflow Sessions so by calling a new process that
	trains over its assigned configurations, a new Session is created automatically.

	'''
	with tf.device('/gpu:' + sys.argv[1]):
		new_index = 326

		pool_arguments_list = []

		a, new_index = get_encoder_decoder_parameters_0(
			new_index,
			lambda ind, lv_c, dr_v:\
				str(ind)\
				+ ': latent_v_count=' + str(lv_c)\
				+ ' dropout_value=' + str(dr_v)
		)
		pool_arguments_list += a

		print( new_index )

		a_list = [\
			list(),\
			list(),\
			list(),\
			list(),\
		]

		i = 0
		for args in pool_arguments_list:
			a_list[i].append(args)
			i = (i + 1) % 4

		global data_set_dict
		data_set_dict = dict()

		encoder_decoder_dict = pickle.load(open(
			'pca_data_sets/7.p',
			'rb'
		))
		encoder_decoder_dict['non_split_data_set'] = np.array( encoder_decoder_dict['non_split_data_set'] )
		data_set_dict['enc_dec_set'] = encoder_decoder_dict

		for get_model_f, get_model_d, model_fit_f, fit_model_d in a_list[int(sys.argv[1])]:

			model = get_model_f(get_model_d)

			model_fit_f( model , fit_model_d )

def launch_train_per_process(ind):
	'''

	DEPRECATED

	Was used to do a grid search in the hyperparameter space. This is part of a
	suboptimal approach. One can not train in parallel multiple models without
	creating new tensorflow Sessions so by calling a new process that
	trains over its assigned configurations, a new Session is created automatically.

	'''
	gpu_string = proc_q.get()

	try:

		print('\n\n\n\nWill start training for index ' + str(ind) + ' on gpu ' + gpu_string + '\n\n\n\n')

		os.system(
			'python3 encoder_decoder_main.py ' + str(ind) + ' ' + gpu_string
		)

		print('\n\n\n\nFinished training for index ' + str(ind) + ' on gpu ' + gpu_string + '\n\n\n\n')

	except:

		err_string = '\n\n\n\nFailed for index ' + str(ind) + ' on gpu ' + gpu_string + '\n\n\n\n'

		print(err_string)

		with open('econder_decoder_errors.txt','a') as f:
			f.write(err_string)

	proc_q.put(gpu_string)

def train_main_for_generator_grid_search():
	'''

	DEPRECATED

	Was used to do a grid search in the hyperparameter space. This is part of a
	suboptimal approach. One can not train in parallel multiple models without
	creating new tensorflow Sessions so by calling a new process that
	trains over its assigned configurations, a new Session is created automatically.

	'''
	print('Will execute',sys.argv,'!')

	new_index = 335

	pool_arguments_list, new_index =\
		get_encoder_decoder_parameters_1(
			new_index,
			lambda ind, lv_c, dr_v, reg_flag:\
				str(ind)\
				+ ': latent_v_count=' + str(lv_c)\
				+ ' dropout_value=' + str(dr_v)\
				+ ' are_regs_used=' + str(reg_flag)\
				+ ' best_arch_from_pca'
		)

	print( 'Last index is' , new_index )

	exclusion_iterable = tuple( range(335, 340) )

	if sys.argv[1] == '-1':

		file = open('econder_decoder_errors.txt','wt')

		file.close()

		available_gpu_tuple = (\
			'/gpu:0',
			'/gpu:1',
			'/gpu:2',
			'/gpu:3',
		)

		global proc_q
		proc_q = Queue()
		for gpu_string in available_gpu_tuple:
			proc_q.put( gpu_string )

		print('Will start process Pool !')

		Pool(len(available_gpu_tuple)).map(
			launch_train_per_process,
			tuple(
				filter(
					lambda ind: pool_arguments_list[ind][1]['new_index'] not in exclusion_iterable,
					range(len(pool_arguments_list))
				)
			)
		)

	else:

		ind = int( sys.argv[1] )

		if pool_arguments_list[ind][1]['new_index'] not in exclusion_iterable:

			with tf.device(sys.argv[2]):

				global data_set_dict

				data_set_dict = dict()

				encoder_decoder_dict = pickle.load(open(
					'pca_data_sets/7.p',
					'rb'
				))
				encoder_decoder_dict['non_split_data_set'] = np.array( encoder_decoder_dict['non_split_data_set'] )
				data_set_dict['enc_dec_set'] = encoder_decoder_dict

				model = pool_arguments_list[ind][0]( pool_arguments_list[ind][1] )

				pool_arguments_list[ind][2]( model , pool_arguments_list[ind][3] )
		else:
			print('\n\n\n\nSkipped for: ' + str(pool_arguments_list[ind][1]['new_index']) + '\n\n\n\n')

def launch_train_per_process_0(ind):
	'''
	Trains a configuration per GPU. First off, an available GPU string is
	taken out of a shared queue. Then a new session is created for that GPU.
	The training commences. After the training is finished, the GPU string is
	put back into the queue so another process can use it.
	'''
	gpu_string = proc_q.get()


	with tf.device(gpu_string):
		with tf.Session() as sess:

			K.set_session(sess)

			print('\n\n\n\nWill start training for index ' + str(pool_arguments_list[ind][1]['new_index']) + ' on gpu ' + gpu_string + '\n\n\n\n')

			model = pool_arguments_list[ind][0]( pool_arguments_list[ind][1] )

			pool_arguments_list[ind][2]( model , pool_arguments_list[ind][3] )

			print('\n\n\n\nFinished training for index ' + str(pool_arguments_list[ind][1]['new_index']) + ' on gpu ' + gpu_string + '\n\n\n\n')

	proc_q.put(gpu_string)

def train_main_for_generator_grid_search_0():
	'''
	Launches grid search over a set of hyperparameter configurations
	'''
	global pool_arguments_list, proc_q

	new_index = 380

	if True:
		pool_arguments_list, new_index =\
			get_encoder_decoder_parameters_1(
				new_index,
				lambda ind, lv_c, dr_v, reg_flag:\
					str(ind)\
					+ ': latent_v_count=' + str(lv_c)\
					+ ' no dropout' + str(dr_v)\
					+ ' are_regs_used=' + str(reg_flag)\
					+ ' best_arch_from_pca'
			)
	if False:
		pool_arguments_list, new_index =\
			get_encoder_decoder_parameters_2(
				new_index,
				lambda ind, lv_c, dr_v:\
					str(ind)\
					+ ': latent_v_count=' + str(lv_c)\
					+ ' dropout_value=' + str(dr_v)\
					+ ' extended_set'
			)

	# exclusion_iterable = tuple( range(335, 340) )
	# exclusion_iterable = tuple( range(340, 350) )
	exclusion_iterable = tuple()

	global data_set_dict

	data_set_dict = dict()

	encoder_decoder_dict = pickle.load(open(
		'pca_data_sets/8_4th_june.p',
		'rb'
	))
	encoder_decoder_dict['non_split_data_set'] = np.array( encoder_decoder_dict['non_split_data_set'] )
	data_set_dict['enc_dec_set'] = encoder_decoder_dict

	file = open('econder_decoder_errors.txt','wt')

	file.close()

	available_gpu_tuple = (\
		'/gpu:0',
		'/gpu:1',
		'/gpu:3',
		'/gpu:4',
	)

	proc_q = Queue()
	for gpu_string in available_gpu_tuple:
		proc_q.put( gpu_string )

	print('Will start process Pool !')

	Pool(len(available_gpu_tuple)).map(
		launch_train_per_process_0,
		tuple(
			filter(
				lambda ind: pool_arguments_list[ind][1]['new_index'] not in exclusion_iterable,
				range(len(pool_arguments_list))
			)
		)
	)

def get_biggest_index_model_path(index):
	'''
	Looks into folder containing models for the configuration at "index".

	index
		integer
	'''
	best_tuple = (1,'model_0001.hdf5')

	for model_name in os.listdir('pca_multiple_model_folders/models_' + str(index)):

		model_index = int(model_name[6:10])

		if best_tuple[0] < model_index: best_tuple = ( model_index , model_name, )

	return './pca_multiple_model_folders/models_' + str(index) + '/' + model_name

def get_biggest_index_model_path_by_val_loss(index):
	'''
	DEPRECATED

	Tries to construct the path to dumped model, but the epoch indexes in the
	CSVLogger are different than the ones in the Model Checkpointer. They differ
	by one. (One starts counting at 0 while the other starts counting at 1).
	'''
	g = csv.reader( open( './pca_csv_folder/' + 'losses_' + str( index ) +'.csv' , 'rt' ) )

	val_loss_index = next(g).index('val_loss')

	models_dict = dict()
	for model_name in os.listdir('./pca_multiple_model_folders/models_' + str(index)):
		models_dict[int(model_name[6:10])] = model_name

	print(tuple(models_dict.keys()))

	best_tuple = next(g)
	best_tuple = ( int(best_tuple[0]) , float(best_tuple[val_loss_index]) , )

	for line in g:
		v = float(line[val_loss_index])
		if v < best_tuple[1]:
			best_tuple = (int(line[0]), v,)

	return './pca_multiple_model_folders/models_' + str(index) + '/' + models_dict[best_tuple[0]]

def get_model_path_by_date(index):
	'''
	Returns the model path from a configuration by creation date of the model.
	'''
	a_list = list()

	for fn in os.listdir('./pca_multiple_model_folders/models_' + str(index)):

		a_list.append(
			(
				fn,
				os.stat(
					'./pca_multiple_model_folders/models_' + str(index) + '/' + fn
				).st_mtime
			)
		)

	a_list.sort( key=lambda p: p[1] , reverse=True )

	return './pca_multiple_model_folders/models_' + str(index) + '/' + a_list[0][0]

def dump_encoder_decoder_plot(index):
	'''
	Dumps plots of the loss evolution on training and validation
	data sets.
	'''
	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	fig = plt.gcf()
	fig.set_size_inches(11,8)

	if True:
		encoder_decoder_dict = pickle.load(open(
			'pca_data_sets/7.p',
			'rb'
		))
		encoder_decoder_dict['non_split_data_set'] = np.array( encoder_decoder_dict['non_split_data_set'] )

		x_set, y_set = generate_random_batch_1(
			len( encoder_decoder_dict['valid_indexes'] ),
			40,
			sorted( encoder_decoder_dict['valid_indexes'] ),
			encoder_decoder_dict,
			only_last_flag=True,
			firs_bs_elements_flag=True,
			one_input_flag=False,
		)

		best_index_path =\
			get_model_path_by_date(index)


		model = keras.models.load_model(best_index_path)

		model.summary()

		print('Best model index at', best_index_path)

		print(index, model.evaluate(x_set,y_set))

		predicted_array = model.predict(x_set)

		predicted_array = predicted_array[1]

		plt.plot(
			range( y_set[1].shape[0] ),
			y_set[1][:,0],
			label='Ground Truth'
		)

		plt.plot(
			range( predicted_array.shape[0] ),
			predicted_array[:,0],
			label='Prediction'
		)

		plt.legend()

		plt.xlabel('Index in Data Set')

		plt.ylabel('Normalized Trend')

		plt.savefig(
			'./pca_plots/' + str(index) + '_valid_gt_vs_pred.png'
		)
		plt.clf()

	loss_gen = csv.reader( open( './pca_csv_folder/losses_' + str(index) + '.csv' , 'rt' ) )

	first_line_list = next(loss_gen)

	train_trend_index = first_line_list.index('dense_5_loss')
	valid_trend_index = first_line_list.index('val_dense_5_loss')

	train_matrices_index = first_line_list.index('time_distributed_1_loss')
	valid_matrices_index = first_line_list.index('val_time_distributed_1_loss')

	trend_train_list, trend_valid_list, matrices_train_list, matrices_valid_list =\
		list(), list(), list(), list()

	for line_list in loss_gen:
		if len(line_list) > 2:
			trend_train_list.append( float( line_list[train_trend_index] ) )
			trend_valid_list.append( float( line_list[valid_trend_index] ) )

			matrices_train_list.append( float( line_list[train_matrices_index] ) )
			matrices_valid_list.append( float( line_list[valid_matrices_index] ) )

	cap_f = lambda l:\
		list(\
			map(\
				lambda e: e if e <= 100 else 100,\
				l,\
			)\
		)

	plt.subplot(211)
	plt.plot(
		range(len(trend_train_list)),
		cap_f(trend_train_list),
		label='Trend Train MAPE'
	)
	plt.plot(
		range(len(trend_valid_list)),
		cap_f(trend_valid_list),
		label='Trend Valid MAPE'
	)
	plt.legend()
	plt.ylabel('MAPE')

	plt.subplot(212)
	plt.plot(
		range(len(matrices_train_list)),
		cap_f(matrices_train_list),
		label='Matrices Train MAPE'
	)
	plt.plot(
		range(len(matrices_valid_list)),
		cap_f(matrices_valid_list),
		label='Matrices Valid MAPE'
	)
	plt.legend()

	plt.xlabel('Epoch Index')

	plt.ylabel('MAPE')

	plt.savefig(
		'./pca_plots/' + str(index) + '_losses.png'
	)
	plt.clf()

def one_time_generator_train():
	'''
	Runs one single training configuration.
	'''
	new_index = 361

	data_set_dict = dict()
	encoder_decoder_dict = pickle.load(open(
		'pca_data_sets/8_28_may.p',
		'rb'
	))
	encoder_decoder_dict['non_split_data_set'] = np.array( encoder_decoder_dict['non_split_data_set'] )
	data_set_dict['enc_dec_set'] = encoder_decoder_dict

	def gen_train(batch_size, window_size):
		while True:
			yield generate_random_batch_1(\
				batch_size,
				window_size,
				data_set_dict['enc_dec_set']['train_indexes'],
				data_set_dict['enc_dec_set'],
				only_last_flag=True,
				one_input_flag=True
			)

	def gen_valid(batch_size, window_size):
		while True:
			yield generate_random_batch_1(\
				batch_size,
				window_size,
				data_set_dict['enc_dec_set']['valid_indexes'],
				data_set_dict['enc_dec_set'],
				only_last_flag=True,
				one_input_flag=True,
			)

	if True:
		bins_no = encoder_decoder_dict['non_split_data_set'].shape[-1] - 1

		print('Bins no is', bins_no)

		inp_layer_2 = keras.layers.Input(shape=(40, bins_no,))

		x = keras.layers.TimeDistributed(
			keras.layers.Dense(
				units=10,
				activation='relu',
			)
		)(inp_layer_2)
		last_common_x = keras.layers.BatchNormalization()(x)

		y1 = keras.layers.TimeDistributed(
			keras.layers.Dense(
				units=bins_no,
				activation='tanh'
			)
		)(last_common_x)

		x = keras.layers.TimeDistributed(
			keras.layers.Dense(
				units=10,
				activation='relu',
			    kernel_regularizer=keras.regularizers.l2(1e-4),
				bias_regularizer=keras.regularizers.l2(1e-4),
				activity_regularizer=keras.regularizers.l2(1e-5)
			)
		)(last_common_x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Dropout(0.1)(x)

		x = keras.layers.Bidirectional(
			keras.layers.LSTM(
				units=10,
				return_sequences=True,
			    kernel_regularizer=keras.regularizers.l2(1e-4),
				bias_regularizer=keras.regularizers.l2(1e-4),
				activity_regularizer=keras.regularizers.l2(1e-5)
			)
		)(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Dropout(0.1)(x)

		x = keras.layers.TimeDistributed(
			keras.layers.Dense(
				units=10,
				activation='relu',
			    kernel_regularizer=keras.regularizers.l2(1e-4),
				bias_regularizer=keras.regularizers.l2(1e-4),
				activity_regularizer=keras.regularizers.l2(1e-5)
			)
		)(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Dropout(0.1)(x)

		x = keras.layers.TimeDistributed(
			keras.layers.Dense(
				units=5,
				activation='relu',
			    kernel_regularizer=keras.regularizers.l2(1e-4),
				bias_regularizer=keras.regularizers.l2(1e-4),
				activity_regularizer=keras.regularizers.l2(1e-5)
			)
		)(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Dropout(0.1)(x)

		x = keras.layers.Bidirectional(
			keras.layers.LSTM(
				units=1,
				return_sequences=False,
			    kernel_regularizer=keras.regularizers.l2(1e-4),
				bias_regularizer=keras.regularizers.l2(1e-4),
				activity_regularizer=keras.regularizers.l2(1e-5)
			)
		)(x)
		x = keras.layers.BatchNormalization()(x)

		y2 = keras.layers.Dense(
			units=1,
			activation='sigmoid',
		    kernel_regularizer=keras.regularizers.l2(1e-4),
			bias_regularizer=keras.regularizers.l2(1e-4),
			activity_regularizer=keras.regularizers.l2(1e-5)
		)(x)

		model = keras.models.Model(inputs=inp_layer_2, outputs=[y1,y2,])

		model.compile(
			optimizer=keras.optimizers.Adam(),
			loss='mean_absolute_percentage_error',
			metrics=['mae',]
		)

	model.summary()

	model.fit_generator(
		gen_train(128, 40),
		epochs=700,
		steps_per_epoch=136,
		validation_data=gen_valid(128,40),
		validation_steps=34,
		verbose=2,
		callbacks=[
			keras.callbacks.CSVLogger(
				'pca_csv_folder/losses_' + str(new_index) + '.csv'
			),
			keras.callbacks.ModelCheckpoint(
				'./pca_multiple_model_folders/models_' + str(new_index) + "/model_{epoch:04d}.hdf5",
				monitor='val_loss',
				save_best_only=True
			),
			keras.callbacks.EarlyStopping(
				monitor='val_loss',
				patience=200,
			),
		]
	)

if __name__ == '__main__':
	# train_main_for_generator_grid_search()
	# with tf.device('/gpu:3'):
	# for i in range(335, 340):
	# 	dump_encoder_decoder_plot(i)
	train_main_for_generator_grid_search_0()
	# dump_encoder_decoder_plot(335)
	# test_generator()