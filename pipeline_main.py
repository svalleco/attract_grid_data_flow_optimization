from multiprocessing import Pool
import os
import json
import pickle
from functools import reduce
import binning_main

ANSWERED_QUERIES_FOLDER = '/data/mipopa/answered_query_dump_folder/'

DATASETS_FOLDER = '/optane/mipopa/multiple_clients_pipeline/data_sets/'

WORKING_FOLDER = '/optane/mipopa/multiple_clients_pipeline/'

def split_queries_per_client_main():
	queries_per_first_option_dict = dict()
	for se_name in pickle.load(open(WORKING_FOLDER + 'necessary_farms_set.p','rb')):
		queries_per_first_option_dict[se_name] = list()

	i = 0
	for q_generator in\
		map(
			lambda ffn:\
				map(
					lambda q_line: (q_line[0],q_line[2][0][0],q_line[3],),
					json.load( open(ANSWERED_QUERIES_FOLDER + ffn,'rt') )
				),
			os.listdir(ANSWERED_QUERIES_FOLDER)
		):
		if i % 10 == 0: print(i)
		for tm, fo, rs in q_generator:
			queries_per_first_option_dict[fo].append((tm,rs))
		i+=1

	pickle.dump(
		queries_per_first_option_dict,
		open(
			'./multiple_clients_pipeline/queries_per_first_option_dict.p',
			'wb'
		)
	)

def dump_necessary_farms_set():
	necessary_farms_set = set()

	for se_generator in\
		map(
			lambda q_line: map(lambda p: p[0], q_line[2]),
			json.load(open('first_option_cern.json','rt'))
		):

		necessary_farms_set.update(se_generator)

	print(necessary_farms_set)

	pickle.dump(
		necessary_farms_set,
		open(
			MISC_FOLDER + 'necessary_farms_set.p',
			'wb'
		)
	)

def analyse_per_proc(i):
	found_true_set = set()

	for emittent_name in map(
			lambda q_line: q_line[1],
			json.load(
				open(ANSWERED_QUERIES_FOLDER + filename_list[i], 'rt')
			)
		):
		to_remove_list = list()
		for se_name in necessary_farms_set:
			if se_name in emittent_name or emittent_name in se_name:
				found_true_set.add(se_name)
				to_remove_list.append(se_name)
		for se_name in to_remove_list:
			necessary_farms_set.remove(se_name)
		if len(necessary_farms_set) == 0: break

	return found_true_set

def analyse_main():
	global filename_list, necessary_farms_set

	filename_list = os.listdir( ANSWERED_QUERIES_FOLDER )

	necessary_farms_set = pickle.load(open(MISC_FOLDER + 'necessary_farms_set.p','rb'))

	found_true_set =\
	reduce(
		lambda acc, x: acc | x,
		Pool(n_proc).map(analyse_per_proc,range(len(filename_list))),
		set()
	)

	print(found_true_set)
	print(necessary_farms_set - found_true_set)

def analyse_main_0():
	print(sorted(
		map(
			lambda name: name.lower(),
			pickle.load(open('first_week_cern_thp_per_client.p','rb'))[1579811802732].keys()
		)
	))
	print(sorted(pickle.load(open(WORKING_FOLDER + 'necessary_farms_set.p','rb'))))

# ['aliendb2.cern.ch', 'bari', 'bratislava', 'catania-vf', 'ccin2p3', 'ccin2p3_2', 'cnaf', 'fzk', 'grenoble', 'grif_ipno', 'gsi', 'hiroshima', 'ihep', 'ipnl', 'iss', 'itep', 'kfki', 'kisti_gsdc', 'kolkata-cream', 'kosice_arc', 'lbl_hpcs', 'legnaro', 'niham', 'pnpi', 'poznan', 'prague_arc', 'rrc-ki', 'saopaulo', 'spbsu', 'subatech', 'sut', 'torino', 'trieste', 'troitsk', 'upb']
# ['bari', 'birmingham', 'bitp', 'bratislava', 'catania', 'ccin2p3', 'cern', 'clermont', 'cnaf', 'cyfronet', 'fzk', 'grenoble', 'grif_ipno', 'grif_irfu', 'gsi', 'hiroshima', 'icm', 'ihep', 'ipnl', 'iss', 'itep', 'jinr', 'kfki', 'kisti_gsdc', 'kolkata', 'kosice', 'lbl_hpcs', 'legnaro', 'mephi', 'ndgf', 'niham', 'nipne', 'ornl', 'pnpi', 'poznan', 'prague', 'ral', 'rrc-ki', 'rrc_ki_t1', 'saopaulo', 'sara', 'sarfti', 'snic', 'spbsu', 'strasbourg_ires', 'subatech', 'sut', 'torino', 'trieste', 'troitsk', 'unam_t1', 'upb', 'za_chpc']

def produce_throughput_dict():
	new_dict = dict()

	for thp_se_name, tm_dict in pickle.load(open('first_week_Se_name_Tm_From_name_dict.p','rb')).items():
		new_dict[thp_se_name] =\
		sorted(
			map(
				lambda p:\
					(
						p[0],
						sum( p[1].values() )
					),
				tm_dict.items()
			)
		)

	pickle.dump(
		new_dict,
		open(WORKING_FOLDER + 'first_week_thp_list_per_reporting_ses.p','wb')
	)

def produce_data_sets_main():

	queries_per_client_dict = pickle.load(open(WORKING_FOLDER+'queries_per_first_option_dict.p','rb'))

	pool_list = list()

	for thp_se_name, thp_list in pickle.load(
		open(WORKING_FOLDER + 'first_week_thp_list_per_reporting_ses.p','rb')).items():

		for que_se_name, q_list in queries_per_client_dict.items():

			if thp_se_name in que_se_name or que_se_name in thp_se_name:

				# pool_list.append(

				# 	binning_main.get_five_minute_binned_dataset_5(
				# 		sorted(q_list),
				# 		thp_list,
				# 		DATASETS_FOLDER + thp_se_name + '.json',
				# 		1579264801390,
				# 		1579875041000,
				# 		95,
				# 	)

				# )

				print(str(len(q_list)) + ' ' + str(len(thp_list)))

				break

	# for p in pool_list:

	# 	p.join()


if __name__ == '__main__':
	global n_proc

	n_proc = 95

	produce_data_sets_main()

	# {'ornl', 'grif_ipno', 'trieste', 'birmingham', 'gsi', 'troitsk', 'poznan', 'bari', 'strasbourg_ires', 'grenoble', 'ral', 'itep', 'cern', 'bratislava', 'rrc-ki', 'kolkata', 'cyfronet', 'catania', 'lbl_hpcs', 'nipne', 'sara', 'spbsu', 'torino', 'ndgf', 'sarfti', 'kosice', 'subatech', 'ihep', 'ccin2p3', 'legnaro', 'kfki', 'unam_t1', 'saopaulo', 'prague', 'bitp', 'sut', 'icm', 'upb', 'iss', 'niham', 'kisti_gsdc', 'snic', 'jinr', 'ipnl', 'mephi', 'fzk', 'cnaf', 'hiroshima', 'clermont', 'grif_irfu', 'za_chpc', 'pnpi', 'rrc_ki_t1'},
