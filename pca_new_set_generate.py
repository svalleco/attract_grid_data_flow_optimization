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

MATRIXES_FOLDER_LIST = (\
	'./matrices_folder/remote_host_0/log_folder/',\
	'./matrices_folder/remote_host_1/log_folder/',\
	'./matrices_folder/remote_host_1/log_folder/',\
)

WEEK_TIME_MOMENTS = (\
	(1579215600000, 1579875041000),\
	(1580511600000, 1581289199000),\
	(1581289200000, 1581980399000),\
)

# vvvv     Third week is shortened     vvvv
# vvvv                                 vvvv
# WEEK_TIME_MOMENTS = (\
# 	(1579215600000, 1579875041000),\
# 	(1580511600000, 1581289199000),\
# 	(1581289200000, 1581670035995),\
# )

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

def dump_new_set():

	# Set up throughput
	thp_iterable_tuple = (
		get_thp('./throughput_folder/january_month_throughput.csv',WEEK_TIME_MOMENTS[0]),
		get_thp('./throughput_folder/february_month.csv',WEEK_TIME_MOMENTS[1]),
		get_thp('./throughput_folder/february_month.csv',WEEK_TIME_MOMENTS[2])
	)
	print('Finished throughput set up !')

	# Set up all read_size_per_time_moment
	read_size_iterable_tuple = sorted(pickle.load( open( PCA_DUMP_FOLDER + 'all_queries_read_size.p', 'rb' )))
	print('Finished read size per time moment set up !')

	get_matrix_f =\
	lambda folder_name, keyword, time_moments, extraction_f:\
		tuple(\
			map(\
				lambda time_tag: ( time_tag , extraction_f( time_tag , folder_name )  ),\
				sorted(\
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
			)\
		)

	# Set up distance matrices
	distance_iterable_tuple =\
	(
		get_matrix_f( MATRIXES_FOLDER_LIST[0] , 'distance' , WEEK_TIME_MOMENTS[0] , get_dist_dict_by_time ),
		get_matrix_f( MATRIXES_FOLDER_LIST[1] , 'distance' , WEEK_TIME_MOMENTS[1] , get_dist_dict_by_time ),
		get_matrix_f( MATRIXES_FOLDER_LIST[2] , 'distance' , WEEK_TIME_MOMENTS[2] , get_dist_dict_by_time ),
	)
	print('Finished distance matrices set up !')

	# Set up demotion matrices
	demotion_iterable_tuple =\
	(
		get_matrix_f( MATRIXES_FOLDER_LIST[0] , 'demotion' , WEEK_TIME_MOMENTS[0] , get_dem_dict_by_time ),
		get_matrix_f( MATRIXES_FOLDER_LIST[1] , 'demotion' , WEEK_TIME_MOMENTS[1] , get_dem_dict_by_time ),
		get_matrix_f( MATRIXES_FOLDER_LIST[2] , 'demotion' , WEEK_TIME_MOMENTS[2] , get_dem_dict_by_time ),
	)
	print('Finished distance matrices set up !')

	# Distance and Demotion Correction - TO DO
	# new_dist_per_week_list, new_dem_per_week_list = list(), list()
	# for old_dist_tuple, old_dem_tuple in zip( distance_iterable_tuple , demotion_iterable_tuple ):
	# 	if old_dist_tuple[0][0] < old_dem_tuple[0][0]:
	# 		ind = 0
	# 		while old_dist_tuple[ind][0] <= old_dem_tuple[0][0]: ind+=1
	# 		ind -= 1

	# 		if old_dist_tuple[ind][0] == old_dem_tuple[0][0]:
	# 			new_dist_tuple = old_dist_tuple[ ind ][0]
	# 			new_dem_tuple = old_dem_tuple
	# 		else:
	# 			pass

	data_set_list = list()

	for i in range(3):
		data_set_list.append(
			create_data_set(
				thp_iterable_tuple[i],
				read_size_iterable_tuple[i],
				distance_iterable_tuple[i],
				demotion_iterable_tuple[i],
			)
		)

	print( len( data_set_list[0] ) )
	print( len( data_set_list[1] ) )
	print( len( data_set_list[2] ) )
	print( 'Total is ' + str( len( data_set_list[0] ) + len( data_set_list[1] ) + len( data_set_list[2] ) ) )

	pickle.dump(
		data_set_list,
		open( PCA_DUMP_FOLDER + 'tm_thp_ars_dist_dem_per_week_4_may.p' , 'wb' )
	)

def get_whole_matrixes():
	data_set_list = list()

	for week_list in pickle.load(open(PCA_DUMP_FOLDER + 'tm_thp_ars_dist_dem_per_week_4_may.p','rb')):

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
		(
			list(map(lambda e: e[-1], data_set_list[0])),
			list(map(lambda e: e[-1], data_set_list[1])),
			list(map(lambda e: e[-1], data_set_list[2])),
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
		open( PCA_DUMP_FOLDER + 'tm_thp_ars_whole_per_week_4_may.p' , 'wb' )
	)

def get_principal_components():
	data_set_list = pickle.load(open(PCA_DUMP_FOLDER + 'tm_thp_ars_whole_per_week_4_may.p','rb'))

	# print( len( data_set_list[0][0][-1] ) )

	whole_matrix_array = np.array(
		list(map(lambda e: e[-1], data_set_list[0]))\
		+ list(map(lambda e: e[-1], data_set_list[1]))\
		+ list(map(lambda e: e[-1], data_set_list[2]))\
	)

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

	if True:
		dump_cell_diff(whole_matrix_array)

	pca_engine = PCA(20)

	pca_engine.fit(
		whole_matrix_array
	)

	print(pca_engine.explained_variance_ratio_.cumsum())

	if True:
		dump_cell_diff(
			pca_engine.transform( whole_matrix_array )[:,:11],
			'pipe_pc_cell_diff_4_may.p'
		)

def analyse_main_0():
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

if __name__ == '__main__':

	# dump_new_set()

	# analyse_main_0()

	get_principal_components()