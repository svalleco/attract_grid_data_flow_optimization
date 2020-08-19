#!/usr/bin/python3
from os import listdir
from sys import argv
from csv import reader
import traceback

'''
File contains a monitoring tool for the grid search. It reads from
what the Keras CSVLogger callback writes to disk.
'''

losses_set = set( listdir( './pca_csv_folder/' ) )

models_folder_set = set( listdir( './pca_multiple_model_folders/' ) )

def get_info_string(index):

	desc_string = ' \''

	if str(index) + '.txt' in listdir( './pca_index_meaning/' ):

		desc_file = open( './pca_index_meaning/' + str(index) + '.txt' , 'rt' )

		desc_string = desc_string + desc_file.read()

		desc_file.close()

	desc_string = desc_string + '\''

	a_s = str( index ) + ': epochs_count='

	if 'losses_' + str( index ) +'.csv' in losses_set:
		a_s += str( len( list( reader( open( './pca_csv_folder/' + 'losses_' + str( index ) +'.csv' , 'rt' ) ) ) ) - 1 )
	else:
		a_s += '0'

	a_s += ' best_model_at='
	# a_s += ' '

	models_dir_path = 'models_' + str(index)

	if models_dir_path in models_folder_set:

		models_names_list = listdir( './pca_multiple_model_folders/' + models_dir_path )

		if len( models_names_list )  == 0:

			return a_s + '0'

		else:

			g = reader( open( './pca_csv_folder/' + 'losses_' + str( index ) +'.csv' , 'rt' ) )

			val_loss_index = next(g).index('val_loss')

			best_valid_model_and_loss = next(g)

			best_valid_model_and_loss = (\
				int(best_valid_model_and_loss[0]),
				float(best_valid_model_and_loss[val_loss_index]),
			)

			for line_list in g:
				a = int(line_list[0])

				val_loss = float(line_list[val_loss_index])

				if val_loss < best_valid_model_and_loss[1]:
					best_valid_model_and_loss = (
						a,
						val_loss
					)

			return a_s + str(best_valid_model_and_loss[0])\
				+ ' best_val_mape=' + str(best_valid_model_and_loss[1])\
				+ desc_string

	return a_s + '0' + desc_string

for t in range(int(argv[1]),\
	int(argv[2]) if argv[2] != '-1' else max( map( lambda dirn: int(dirn.split('_')[-1]) ,\
	filter( lambda dirn: 'models_' in dirn , listdir('./pca_multiple_model_folders/') ) ) )):
	try:
		print(get_info_string(t))
	except Exception as exc:
		print(str(t)+':',traceback.format_exc(),exc)

# with open('pca_what_is_what.txt', 'wt') as myfile:
# 	for _, content in sorted( map(
# 		lambda fn: ( int(fn[:-4]) ,  open('pca_index_meaning/'+fn).read() ,),
# 		listdir( 'pca_index_meaning' ) ) ):
# 			myfile.write(content + '\n')