from urllib.request import urlopen
from bs4 import BeautifulSoup
import os
import time
import pickle
import random
from functools import reduce
from collections import namedtuple
from copy import deepcopy
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

ClientInfo = namedtuple('ClientInfo', ['probs_per_SE_list', 'StElemInfo_per_SE_list'])

StElemInfo = namedtuple('StElemInfo', ['se_name_tuple', 'locality_bool', 'se_index',])

class Mapping:
	def __init__(self):
		self.__next_client_index = 0
		self.__next_se_index = 0
		self.__client_dict = dict()
		self.__se_dict = dict()

	def add_clients(self, cl_iterable):
		for cl in cl_iterable:
			if cl not in self.__client_dict:
				self.__client_dict[cl] = self.__next_client_index
				self.__next_client_index += 1

	def add_ses(self, se_iterable):
		for se in se_iterable:
			if se not in self.__se_dict:
				self.__se_dict[se] = self.__next_se_index
				self.__next_se_index += 1

	def get_client_index(self, name):
		return self.__client_dict[name]

	def get_se_index(self, name):
		return self.__se_dict[name]

	def get_max_indexes(self):
		return self.__next_client_index-1, self.__next_se_index-1,

	def dump_mapping(self):
		pickle.dump(
			(
				self.__next_client_index,
				self.__next_se_index,
				self.__client_dict,
				self.__se_dict,
			),
			# open('./mappings/' + str(int(round(time.time() * 1000))) + '.p','wb')
			open('./mappings/m.p','wb')
		)

	def load_mapping(self, filepath):
		self.__next_client_index,\
		self.__next_se_index,\
		self.__client_dict,\
		self.__se_dict = pickle.load(open(filepath,'rb'))

	def print_mapping(self):
		for key, value in self.__client_dict.items():
			print(key + ' ' + str(value))
		print()
		for key, value in self.__se_dict.items():
			print(str(key) + ' ' + str(value))

class QueryStatistic:
	def __init__(self, day_index):
		'''
			relevant_from:
				stores time moment in miliseconds when the statistics is being relevant

			__local_read_size_per_SE_dict:
				the average sizes of files read locally per SE

			__remote_read_size_per_SE_dict:
				the average sizes of files read remotely per SE

			__clients_names_tuple
				names of clients in the order present in the html file

			__client_weights_list
				weights of the clients in the order present in the html file

			__client_info_dict
				{ client_name : ( probability_to_communicate_per_SE , StElemInfo ) }
		'''
		self.day_index = day_index

		self.relevant_from = int(round(time.time() * 1000)) - day_index * 86400000

		soup = BeautifulSoup(urlopen('http://alimonitor.cern.ch/iostat.jsp?pid=-' + str(day_index)).read().decode('UTF-8'),features="lxml")

		self.__se_names_tuple = tuple(
			map(
				lambda e: (str(e.contents[0]).lower(), str(e.contents[2][:-1]\
					).lower(),) if len(e.contents) == 3 else (str(e.contents[0][:-1]).lower(),),
				soup.body.thead.contents[3].contents[6:]
			)
		)

		self.total_number_of_files_read = int(soup.body.tfoot.contents[1].contents[2].contents[0].split(' ')[0])

		remote_i = 4

		local_list = []
		local_total_list = []
		local_count_list = []

		remote_list = []
		remote_total_list = []
		remote_count_list = []

		for local_el in soup.body.tfoot.contents[1].contents[7:]:

			if local_el != '\n':

				if len(local_el.get_text()) < 10:

					local_list.append(0)

					local_total_list.append(0)

					local_count_list.append(0)

				else:

					a, b = local_el.get_text().split('B/s')[1].split(' ')

					a = float(a)

					if 'PB' in b:
						a *= 1125899906842624
					elif 'TB' in b:
						a *= 1099511627776
					else:
						a *= 1073741824

					local_total_list.append(a)

					local_count_list.append(int(local_el.get_text().split(' (')[0].split('\n')[1]))

					local_list.append( local_total_list[-1] / local_count_list[-1] )


			if soup.body.tfoot.contents[3].contents[remote_i] != '\n':

				if len(soup.body.tfoot.contents[3].contents[remote_i].get_text()) < 10:

					remote_list.append(0)

					remote_total_list.append(0)

					remote_count_list.append(0)

				else:
					a, b = soup.body.tfoot.contents[3].contents[remote_i\
						].get_text().split('B/s')[1].split(' ')

					a = float(a)

					if 'PB' in b:
						a *= 1125899906842624
					elif 'TB' in b:
						a *= 1099511627776
					else:
						a *= 1073741824

					remote_total_list.append(a)

					remote_count_list.append(int(soup.body.tfoot.contents[3].contents[\
						remote_i].get_text().split(' (')[0].split('\n')[1]))

					remote_list.append( remote_total_list[-1] / remote_count_list[-1] )

			remote_i += 1

		self.__local_read_size_per_SE_dict = dict()
		self.__remote_read_size_per_SE_dict = dict()
		self.__local_total_per_SE_dict = dict()
		self.__local_count_per_SE_dict = dict()
		self.__remote_total_per_SE_dict = dict()
		self.__remote_count_per_SE_dict = dict()
		for i in range(len(self.__se_names_tuple)):
			self.__local_read_size_per_SE_dict[\
				self.__se_names_tuple[i]] = local_list[i]
			self.__remote_read_size_per_SE_dict[\
				self.__se_names_tuple[i]] = remote_list[i]
			self.__local_total_per_SE_dict[\
				self.__se_names_tuple[i]] = local_total_list[i]
			self.__local_count_per_SE_dict[\
				self.__se_names_tuple[i]] = local_count_list[i]
			self.__remote_total_per_SE_dict[\
				self.__se_names_tuple[i]] = remote_total_list[i]
			self.__remote_count_per_SE_dict[\
				self.__se_names_tuple[i]] = remote_count_list[i]

		self.__clients_names_tuple = []

		self.__client_weights_list = []

		self.__client_info_dict = dict()

		for l in soup.body.tbody.contents:
			if len(l) > 1:

				self.__clients_names_tuple.append(\
					str(l.contents[1].contents[1].get_text()).lower()
				)

				percentages_per_SE_list = []

				locality_per_SE_list = []

				i = 0

				for c in l.contents[10:]:

					if c != '\n':

						if c.get_text() == '\xa0\n':

							percentages_per_SE_list.append( 0 )

							locality_per_SE_list.append(StElemInfo(
								self.__se_names_tuple[i],
								False,
								i,
							))

						else:

							percentages_per_SE_list.append( float(c.get_text().split('(')[1].split('%')[0])/100 )

							locality_per_SE_list.append(StElemInfo(
								self.__se_names_tuple[i],
								len(c.contents) == 2,
								i,
							))

						i += 1

				self.__client_weights_list.append(\
					int(l.contents[4].get_text().split(' files')[0])
				)

				self.__client_info_dict[self.__clients_names_tuple[-1]] = ClientInfo(\
					percentages_per_SE_list,\
					locality_per_SE_list,\
				)

		self.__clients_names_tuple = tuple(self.__clients_names_tuple)

	def generate_queries(self, query_times_tuple):
		'''
		Generates queries based on the statistics: [(time, client_name, (storage elements), read size)]
		'''
		local_eliminate_by_index = lambda a,se_list:\
		tuple(\
			map(\
				lambda p: p[1],
				filter(
					lambda p: p[0] not in map(lambda pp: pp.se_index, se_list),
					enumerate(a)
				)
			)
		)

		q_list = []
		for client_name, time_moment in zip(random.choices( self.__clients_names_tuple ,\
			self.__client_weights_list , k=len(query_times_tuple)), query_times_tuple):

			storage_elements_list = random.choices(\
				self.__client_info_dict[client_name].StElemInfo_per_SE_list,
				self.__client_info_dict[client_name].probs_per_SE_list,
			)

			storage_elements_list += random.choices(\
				local_eliminate_by_index(\
					self.__client_info_dict[client_name].StElemInfo_per_SE_list,
					storage_elements_list,
				),
				local_eliminate_by_index(\
					self.__client_info_dict[client_name].probs_per_SE_list,
					storage_elements_list,
				),
			)

			storage_elements_list += random.choices(\
				local_eliminate_by_index(\
					self.__client_info_dict[client_name].StElemInfo_per_SE_list,
					storage_elements_list,
				),
				local_eliminate_by_index(\
					self.__client_info_dict[client_name].probs_per_SE_list,
					storage_elements_list,
				),
			)

			if False:
				for se in storage_elements_list:
					c = 0
					for se1 in storage_elements_list:
						if se.se_name_tuple == se1.se_name_tuple:
							c+=1
						if c == 2:
							print(storage_elements_list)
							exit(0)

			if False:
				if len(storage_elements_list) != 3:
					print(storage_elements_list)
					exit(0)

			read_size_tuple = tuple(map(
				lambda sei: self.__local_read_size_per_SE_dict[\
					sei.se_name_tuple] if sei.locality_bool else\
					self.__remote_read_size_per_SE_dict[sei.se_name_tuple],
				storage_elements_list
			))

			q_list.append((
				time_moment,
				client_name,
				tuple(map(lambda sei: sei.se_name_tuple, storage_elements_list)),
				sum(read_size_tuple) / len(read_size_tuple)
			))

		return q_list

	def eliminate_client(self, client_name):
		'''
		Eliminates a client.
		'''
		client_index = self.__clients_names_tuple.index(client_name)

		self.total_number_of_files_read =\
			self.total_number_of_files_read\
			- self.__client_weights_list[client_index]

		self.__clients_names_tuple =\
			self.__clients_names_tuple[:client_index]\
			+ self.__clients_names_tuple[client_index + 1:]

		self.__client_weights_list =\
			self.__client_weights_list[:client_index]\
			+ self.__client_weights_list[client_index + 1:]

		del self.__client_info_dict[client_name]

	def eliminate_storage_element(self, se_name_tuple):
		'''
		Eliminate a storage element.
		'''
		self.total_number_of_files_read =\
			self.total_number_of_files_read\
			- self.__local_count_per_SE_dict[se_name_tuple]\
			- self.__remote_count_per_SE_dict[se_name_tuple]

		del self.__local_read_size_per_SE_dict[se_name_tuple]

		del self.__local_total_per_SE_dict[se_name_tuple]

		del self.__local_count_per_SE_dict[se_name_tuple]

		del self.__remote_read_size_per_SE_dict[se_name_tuple]

		del self.__remote_total_per_SE_dict[se_name_tuple]

		del self.__remote_count_per_SE_dict[se_name_tuple]

		i = self.__se_names_tuple.index(se_name_tuple)

		self.__se_names_tuple = self.__se_names_tuple[:i] +\
			self.__se_names_tuple[i+1:]

		for _ , value in self.__client_info_dict.items():
			for j in range(i+1,len(value.StElemInfo_per_SE_list)):
				value.StElemInfo_per_SE_list[j] = StElemInfo(
					value.StElemInfo_per_SE_list[j].se_name_tuple,
					value.StElemInfo_per_SE_list[j].locality_bool,
					value.StElemInfo_per_SE_list[j].se_index - 1,
				)
			del value.probs_per_SE_list[i]
			del value.StElemInfo_per_SE_list[i]

	def generate_average_query_statistic(self, other_qs):
		'''
		Generates a average statistic.
		'''
		avg_qs = deepcopy(self)

		for se_tuple in avg_qs.__local_read_size_per_SE_dict.keys():
			if se_tuple in other_qs.__local_read_size_per_SE_dict:
				avg_qs.__local_read_size_per_SE_dict[se_tuple] =\
					(avg_qs.__local_read_size_per_SE_dict[se_tuple]\
					+ other_qs.__local_read_size_per_SE_dict[se_tuple])/2
				avg_qs.__remote_read_size_per_SE_dict[se_tuple] =\
					(avg_qs.__remote_read_size_per_SE_dict[se_tuple]\
					+ other_qs.__remote_read_size_per_SE_dict[se_tuple])/2

		for i, cl_name in enumerate( avg_qs.__clients_names_tuple ):
			if cl_name in other_qs.__clients_names_tuple:
				avg_qs.__client_weights_list[i] = (\
					avg_qs.__client_weights_list[i] +\
					other_qs.__client_weights_list[\
						other_qs.__clients_names_tuple.index(cl_name)
					]\
				)/2
				for j, se_names in enumerate( avg_qs.__se_names_tuple ):
					if se_names in other_qs.__se_names_tuple:
						avg_qs.__client_info_dict[cl_name].probs_per_SE_list[j] =\
						(avg_qs.__client_info_dict[cl_name].probs_per_SE_list[j] +\
						other_qs.__client_info_dict[cl_name].probs_per_SE_list[\
						other_qs.__se_names_tuple.index(se_names)]\
						)/2
		return avg_qs

	def generate_single_day_statistic(self, other_qs):
		avg_qs = deepcopy(self)

		for se_tuple in filter(lambda se: se in other_qs.__se_names_tuple,\
			avg_qs.__se_names_tuple):

			if avg_qs.__local_count_per_SE_dict[se_tuple] != 0:
				avg_qs.__local_total_per_SE_dict[se_tuple] =\
					avg_qs.__local_total_per_SE_dict[se_tuple] - other_qs.__local_total_per_SE_dict[se_tuple]
				avg_qs.__local_count_per_SE_dict[se_tuple] =\
					avg_qs.__local_count_per_SE_dict[se_tuple] - other_qs.__local_count_per_SE_dict[se_tuple]
				if avg_qs.__local_count_per_SE_dict[se_tuple] != 0:
					avg_qs.__local_read_size_per_SE_dict[se_tuple]=\
						avg_qs.__local_total_per_SE_dict[se_tuple] / avg_qs.__local_count_per_SE_dict[se_tuple]
				else:
					avg_qs.__local_read_size_per_SE_dict[se_tuple] = 0

			if avg_qs.__remote_count_per_SE_dict[se_tuple] != 0:
				avg_qs.__remote_total_per_SE_dict[se_tuple] =\
					avg_qs.__remote_total_per_SE_dict[se_tuple] - other_qs.__remote_total_per_SE_dict[se_tuple]
				avg_qs.__remote_count_per_SE_dict[se_tuple] =\
					avg_qs.__remote_count_per_SE_dict[se_tuple] - other_qs.__remote_count_per_SE_dict[se_tuple]
				if avg_qs.__remote_count_per_SE_dict[se_tuple] != 0:
					avg_qs.__remote_read_size_per_SE_dict[se_tuple]=\
						avg_qs.__remote_total_per_SE_dict[se_tuple] / avg_qs.__remote_count_per_SE_dict[se_tuple]
				else:
						avg_qs.__remote_read_size_per_SE_dict[se_tuple] = 0

		avg_qs.total_number_of_files_read = avg_qs.total_number_of_files_read\
			- other_qs.total_number_of_files_read

		for i, cl_name in enumerate( avg_qs.__clients_names_tuple ):
			if cl_name in other_qs.__clients_names_tuple:
				avg_qs.__client_weights_list[i] =\
					avg_qs.__client_weights_list[i] -\
					other_qs.__client_weights_list[\
						other_qs.__clients_names_tuple.index(cl_name)
					]
				for j, se_names in enumerate( avg_qs.__se_names_tuple ):
					if se_names in other_qs.__se_names_tuple:
						avg_qs.__client_info_dict[cl_name].probs_per_SE_list[j] =\
						avg_qs.__client_info_dict[cl_name].probs_per_SE_list[j] -\
						other_qs.__client_info_dict[cl_name].probs_per_SE_list[\
						other_qs.__se_names_tuple.index(se_names)]
		return avg_qs

	def eliminate_clients_and_se_which_are_not_in(self, client_name_set, se_name_set):
		'''
		Eliminates clients and name sets which are not in the sets given as arguments.
		'''
		cl_to_elim_list = []
		for cl in self.__clients_names_tuple:
			if cl not in client_name_set:
				cl_to_elim_list.append(cl)

		for cl in cl_to_elim_list:
			self.eliminate_client(cl)

		cl_to_elim_list = []
		for cl in self.__se_names_tuple:
			if cl not in se_name_set:
				cl_to_elim_list.append(cl)

		for cl in cl_to_elim_list:
			self.eliminate_storage_element(cl)

	def add_to_mapping(self, mapping):
		mapping.add_clients(self.__clients_names_tuple)
		mapping.add_ses(self.__se_names_tuple)

	def debug_print(self):
		# print(self.__client_weights_list[0])
		print(self.total_number_of_files_read)

def dump_statistic_per_proc(i):
	previous_one_qs = QueryStatistic(i)
	previous_one_qs.eliminate_clients_and_se_which_are_not_in(\
		g_client_name_set,g_se_name_set)

	return i,previous_one_qs

def dump_statistics_between(first_moment, last_moment):
	'''
	Dumps statiscs between first_day and last_day.
	'''
	print('Will start looking for first day !')

	current_time = int(round(time.time() * 1000))

	dump_folder = './statistics_dump/' + str(current_time) + '/'

	os.mkdir(dump_folder)

	i = 0

	time_iter = current_time

	while time_iter > first_moment:

		time_iter -= 86400000

		i+=1

	day_before_first_tuple = (i, time_iter,)

	print('Will start looking for last day !')

	i = 0

	time_iter = current_time

	while time_iter > last_moment:

		time_iter -= 86400000

		i+=1

	day_after_last_tuple = (i - 1, time_iter + 86400000,)

	print(f'Will dump between {day_before_first_tuple[0]} and {day_after_last_tuple[0] - 1} !')

	global g_client_name_set, g_se_name_set

	for fn in os.listdir('./remote_host/log_folder/'):
		if '_distance' in fn:
			content = open('./remote_host/log_folder/' + fn,'rt').read().split('\n')[1:-1]
			if 7238 == len(content):
				g_client_name_set = set()
				g_se_name_set = set()
				for client, storage in map(
					lambda e: (e[0].lower(), (e[1][0].lower(),e[1][1].lower(),)),
					map(lambda e: (e[0], e[1].split('::')[1:],),
					map(lambda r: r.split(';'),content))):
					g_client_name_set.add(client)
					g_se_name_set.add(storage)
				break

	qs_dict = dict(
		Pool(7).map(
			dump_statistic_per_proc,
			range(day_before_first_tuple[0], day_after_last_tuple[0] - 2, -1)
		)
	)

	for i in range(day_before_first_tuple[0] - 1, day_after_last_tuple[0] - 2, -1):
		pickle.dump(\
			qs_dict[i+1].generate_single_day_statistic(qs_dict[i]),
			open(dump_folder + str(i) + '.p', 'wb')
		)

def debug_plot(a_list):
	plt.plot(tuple(range(len(a_list))), tuple(map(lambda e: e[0],a_list)), 'b+')
	plt.show()

def generate_queries_from_dumped_statistics(number_of_queries , first_moment, last_moment,\
	folder_name):
	'''
	Read the statistics and query them.
	'''
	qs_list = list(
		map(
			lambda fn: pickle.load(open(folder_name + fn, 'rb')),
			os.listdir(folder_name)
		)
	)
	qs_list.sort(key=lambda e: e.relevant_from)

	for i in range(len(qs_list)):

		if qs_list[i].relevant_from <= first_moment < qs_list[i].relevant_from + 86400000:

			first_count = round(qs_list[i].total_number_of_files_read *\
				(qs_list[i].relevant_from + 86400000 - first_moment) / 86400000)

			qs_list = qs_list[i:]

			break

	for i in range(len(qs_list)-1, -1, -1):
		if qs_list[i].relevant_from <= last_moment < qs_list[i].relevant_from + 86400000:

			last_count = round(qs_list[i].total_number_of_files_read *\
				(last_moment - qs_list[i].relevant_from) / 86400000)

			qs_list = qs_list[:i+1]

			break

	files_per_qs_list = [first_count,]

	for qs in qs_list[1:len(qs_list)-1]:

		files_per_qs_list.append(qs.total_number_of_files_read)

	files_per_qs_list.append(last_count)

	queries_per_qs_list = list(map(\
		lambda e: round( number_of_queries * e / sum(files_per_qs_list) ),
		files_per_qs_list
	))

	s = sum(queries_per_qs_list)

	if s < number_of_queries:
		for _ in range(number_of_queries - s):
			queries_per_qs_list[random.randint(0,len(queries_per_qs_list)-1)] += 1

	# print(queries_per_qs_list)

	q_list = qs_list[0].generate_queries(\
		tuple(range(
			first_moment,
			qs_list[0].relevant_from + 86400000,
			(qs_list[0].relevant_from + 86400000 - first_moment) // queries_per_qs_list[0]
	)))
	for i in range(1,len(qs_list)-1,1):
		q_list += qs_list[i].generate_queries(\
			tuple(range(qs_list[i].relevant_from, qs_list[i].relevant_from + 86400000, 86400000 // queries_per_qs_list[i]))
		)
	q_list += qs_list[-1].generate_queries(\
		tuple(range(
			qs_list[-1].relevant_from,
			last_moment,
			(last_moment - qs_list[-1].relevant_from) // queries_per_qs_list[-1]
	)))

	s = len(q_list)

	if s > number_of_queries:
		to_elim_set = set()

		for _ in range(s - number_of_queries):
			a = random.randint(0, len(q_list)-1)
			while a in to_elim_set:
				a = random.randint(0, len(q_list)-1)
			to_elim_set.add(a)

		if False:
			ccc = list(
				map(
					lambda e: e[1],
					filter(
						lambda p: p[0] not in to_elim_set,
						enumerate(q_list)
					)
				)
			)

			debug_plot(sorted(ccc))
			print(ccc[0])
			exit(0)

			return ccc

		return list(
			map(
				lambda e: e[1],
				filter(
					lambda p: p[0] not in to_elim_set,
					enumerate(q_list)
				)
			)
		)

	if False:
		debug_plot(sorted(q_list))
		print(q_list[0])
		exit(0)

	return q_list

def answer_queries(query_list, first_moment, last_moment):
	'''
	It generates the answers for queries.
	'''
	filename_tuple = tuple(os.listdir('./remote_host/log_folder/'))
	distance_list = list(
		map(
			lambda e: int(e.split('_')[0]),
			filter(\
				lambda fn: '_distance' in fn,
				filename_tuple
			)
		)
	)
	demotion_list = list(
		map(
			lambda e: int(e.split('_')[0]),
			filter(\
				lambda fn: '_demotion' in fn,
				filename_tuple
			)
		)
	)
	distance_list.sort()
	demotion_list.sort()

	# print(f'{len(distance_list)} {len(demotion_list)}')

	i = len(distance_list)-1
	while distance_list[i] > first_moment: i-=1
	distance_list = distance_list[i:]

	i = len(demotion_list)-1
	while demotion_list[i] > first_moment: i-=1
	demotion_list = demotion_list[i:]

	i = 0
	while i < len(distance_list) and distance_list[i] < last_moment: i+=1
	distance_list = distance_list[:i+1]

	i = 0
	while i < len(demotion_list) and demotion_list[i] < last_moment: i+=1
	demotion_list = demotion_list[:i+1]

	# print(f'{len(distance_list)} {len(demotion_list)}')

	extract_data_function =\
	lambda fn_distance, fn_demotion:\
		(\
			tuple(
				map(
					lambda e: ( e[0].lower() , ( e[1][0].lower() , e[1][1].lower() , ) , e[2] , ),
					map(
						lambda e: (e[0], e[1].split('::')[1:], float(e[2]),),
						map(
							lambda r: r.split(';'),
							open('./remote_host/log_folder/' + fn_distance,'r').read().split('\n')[1:-1]
						)
					)
				)
			),
			tuple(
				map(
					lambda e: ( ( e[0][0].lower() , e[0][1].lower() , ) , e[1] , ),
					map(
						lambda e: (e[0].split('::')[1:], float(e[3]),),
						map(
							lambda r: r.split(';'),
							open('./remote_host/log_folder/' + fn_demotion,'r').read().split('\n')[1:-1]
						)
					)
				)
			),
		)

	dist_i = 0
	dem_i = 0

	answers_list = []

	for time_moment, client_name, storage_elements_list, read_size in query_list:

		if dist_i == len(distance_list) - 2 and time_moment < distance_list[dist_i]:
			dist_i+=1
		else:
			while dist_i < len(distance_list) - 2\
				and not ( distance_list[dist_i]<= time_moment <\
				distance_list[dist_i+1] ):
				dist_i+=1

		if dem_i == len(demotion_list) - 2 and time_moment < demotion_list[dem_i]:
			dem_i+=1
		else:
			while dem_i < len(demotion_list) - 2\
				and not ( demotion_list[dem_i] <= time_moment <\
				demotion_list[dem_i+1] ):
				dem_i+=1

		distance_tuple, demotion_tuple = extract_data_function(\
			str(distance_list[dist_i]) + '_distance',
			str(demotion_list[dem_i]) + '_demotion'
		)

		score_list = []

		for se_names in storage_elements_list:
			for client_it, se_it, value in distance_tuple:
				if client_name == client_it and se_it == se_names:
					score_list.append([\
						se_it,
						value
					])
					break
			for se_it, value in demotion_tuple:
				if se_it == se_names:
					score_list[-1][1] += value
					break

		score_list.sort(key=lambda k: k[1])

		answers_list.append(
			(
				time_moment,
				client_name,
				tuple(
					map(
						lambda e: e[0],
						score_list
					)
				),
				read_size,
			)
		)

	return answers_list

def get_throughput(first_moment, last_moment):
	'''
	Returns a dictionary with all the logged throughput values.
	'''
	spool_dir_path = './remote_host/spool/'

	# get_available_time_moments() --> (1569927214337, 1571212229285)

	filename_list = list(
		filter(
			lambda e: '.done' in e,
			os.listdir(spool_dir_path)
		)
	)

	filename_list.sort()

	for i in range(len(filename_list)):
		content = tuple( map(
			lambda e: int(e[1]),
			filter(
				lambda e: len(e) > 5\
					and '_OUT_freq' not in e[5]\
					and '_IN_freq' not in e[5]\
					and '_IN' not in e[5],
				map(
					lambda line: line.split('\t'),
					open(spool_dir_path+filename_list[i],'rt').read().split('\n')
				)
			))
		)

		if content[-1] >= first_moment:
			filename_list = filename_list[i:]
			break

	i = len(filename_list) - 1
	while i >= 0:
		content = tuple( map(
			lambda e: int(e[1]),
			filter(
				lambda e: len(e) > 5\
					and '_OUT_freq' not in e[5]\
					and '_IN_freq' not in e[5]\
					and '_IN' not in e[5],
				map(
					lambda line: line.split('\t'),
					open(spool_dir_path+filename_list[i],'rt').read().split('\n')
				)
			))
		)

		if content[0] >= last_moment:
			filename_list = filename_list[:i+1]
			break

		i-=1

	site_throughput_dict = dict()

	for fn in filename_list:
		for time_stamp, site_name, tag, value in map(
				lambda e: (int(e[1]), e[2], e[5], float(e[6]),),
				filter(
					lambda e: len(e) == 7\
						and '_OUT_freq' not in e[5]\
						and '_IN_freq' not in e[5],\
					map(
						lambda line: line.split('\t'),
						open(spool_dir_path+fn,'rt').read().split('\n')
					)
				)
			):

			if first_moment <= time_stamp < last_moment + 3600 * 1000:
				if site_name not in site_throughput_dict:
					site_throughput_dict[site_name] = { time_stamp : value }
				else:
					if time_stamp in site_throughput_dict[site_name]:
						site_throughput_dict[site_name][time_stamp] += value
					else:
						site_throughput_dict[site_name][time_stamp] = value

	for site_name in site_throughput_dict.keys():
		site_throughput_dict[site_name] = list(site_throughput_dict[site_name].items())
		site_throughput_dict[site_name].sort()

	# site_throughput_dict := { site_name : [ ( time_stamp , value ) ] }

	for key in tuple(site_throughput_dict.keys()):
		site_throughput_dict[key.lower()] = site_throughput_dict[key]
		del site_throughput_dict[key]

	return site_throughput_dict

def get_associated_set(q_list, thp_list):
	'''
	Couples the query list with the throughput list.
	'''
	association_list = []

	print(len(q_list))

	i = 0

	for q in q_list:
		while i < len(thp_list) and thp_list[i][0] < q[0]:
			i+=1
		association_list.append(q + (thp_list[i][1],))

	return association_list

def get_associated_set_1(q_list, thp_list):
	'''
	Couples the query list with the throughput list.
	'''
	association_list = []

	print(len(q_list))

	for q in q_list:
		if q[0] <= thp_list[0][0]:
			association_list.append(q + (thp_list[0][1],))
		elif q[0] > thp_list[-1][0]:
			association_list.append(q + (thp_list[-1][1],))
		else:
			i = 0
			while not (thp_list[i][0] < q[0] <= thp_list[i+1][0]):
				i+=1
			association_list.append(q + (thp_list[i+1][1],))

	return association_list

def associate_per_proc(q):
	if q[0] <= g_thp_list[0][0]:
		return q + (g_thp_list[0][1],)

	if q[0] > g_thp_list[-1][0]:
		return q + (g_thp_list[-1][1],)

	i = 0
	while not (g_thp_list[i][0] < q[0] <= g_thp_list[i+1][0]):
		i+=1
	return q + (g_thp_list[i+1][1],)

def get_associated_set_2(q_list, thp_list):
	'''
	Couples the query list with the throughput list.
	'''
	global g_thp_list

	g_thp_list = thp_list

	print('Will start answers-throughput association !')

	return Pool(7).map(associate_per_proc, q_list)

def dump_mapping(qs_dump_folder):
	m = Mapping()

	for fn in os.listdir(qs_dump_folder):
		pickle.load(open(qs_dump_folder + fn, 'rb')).add_to_mapping(m)

	m.dump_mapping()

def get_input_set(\
	query_count,\
	first_time_moment,\
	last_time_moment,\
	dump_statistics_folder,\
	mapping_path='./mappings/m.p'\
	):
	m = Mapping()

	m.load_mapping(mapping_path)

	cl_max_num, se_max_num = m.get_max_indexes()

	cl_max_num += 1

	se_max_num += 1

	print(f'max indexes: {cl_max_num} {se_max_num}')

	answers_list = get_associated_set_2(
		answer_queries(
			generate_queries_from_dumped_statistics(
				query_count,
				first_time_moment,
				last_time_moment,
				dump_statistics_folder,
			),
			first_time_moment,
			last_time_moment,
		),
		get_throughput(\
			first_time_moment,
			last_time_moment + 86400000,
		)['grif_ipno'],
	)

	print('Finished generating set !')

	X = np.zeros((query_count, cl_max_num + 3 * se_max_num + 3,))

	for i in range(query_count):
		X[i,m.get_client_index(answers_list[i][1])] = 1
		X[i,cl_max_num+m.get_se_index(answers_list[i][2][0])] = 1
		X[i,cl_max_num+se_max_num+m.get_se_index(answers_list[i][2][1])] = 1
		X[i,cl_max_num+2*se_max_num+m.get_se_index(answers_list[i][2][2])] = 1
		X[i,cl_max_num+3*se_max_num] = answers_list[i][3]
		X[i,cl_max_num+3*se_max_num+1] = answers_list[i][0]
		X[i,cl_max_num+3*se_max_num+2] = answers_list[i][4]

	return X

def shuffle(X):
	l = list(range(X.shape[0]))

	random.shuffle(l)

	new_X = np.empty(X.shape)

	for i,j in enumerate(l):
		new_X[i] = X[j]

	return new_X

def plot_queries_per_throughput_value():
	import matplotlib.pyplot as plt

	X = pickle.load(open('./pipeline_set.p','rb'))

	print(X[:10])

	v_dict = dict()

	for v in X:
		if v not in v_dict:
			v_dict[v] = 1
		else:
			v_dict[v] += 1

	k_list = list(v_dict.keys())
	k_list.sort()

	print(len(k_list))

	plt.plot(k_list, tuple(map(lambda e: v_dict[e], k_list)),'bo')
	plt.show()

def get_moments():
	dist_and_dem_folder = './remote_host/log_folder/'

	filename_list = list(os.listdir(dist_and_dem_folder))

	filename_list.sort()

	print(filename_list[0])
	print(filename_list[-1])

def validate_data_set(X, first_moment, last_moment):
	print(first_moment)
	print(X[0,-2])
	print(X[-1,-2])
	print(last_moment)

if __name__ == '__main__':
	first_time_moment, last_time_moment = 1569927214337, 1574678721605
	dump_statistics_folder = './statistics_dump/1575230096617/'

	if False:
		get_moments()

	if False:
		m = Mapping()

		m.load_mapping('./mappings/m.p')

		print( m.get_max_indexes() )

	if True:
		data_set = get_input_set(
			1000,
			first_time_moment,
			last_time_moment,
			dump_statistics_folder
		)
		# validate_data_set(data_set, first_time_moment, last_time_moment)
		pickle.dump(
			data_set,
			open('./pipe_san.p', 'wb'),
		)

	if False:
		get_input_set(
			2000,
			10,
			first_time_moment,
			last_time_moment,
			dump_statistics_folder,
		)

	if False:
		dump_mapping(dump_statistics_folder)

	if False:
		dump_statistics_between(first_time_moment,last_time_moment)

	if False:
		print(len(generate_queries_from_dumped_statistics(
			20000,
			first_time_moment,
			last_time_moment,
			dump_statistics_folder
		)))

	if False:
		a = get_throughput(first_time_moment, last_time_moment)['grif_ipno']
		print(len(a))

	if False:

		import matplotlib.pyplot as plt

		X = pickle.load(open('pipeline_set.p', 'rb'))[3000:6000]

		max_value = -np.ones(2)

		for l in X:
			max_value = np.maximum(max_value, l[0,-2:])

		x_list, y_list = reduce(
			lambda acc, i: (\
				acc[0] + (2*X[i, 0, -2]/max_value[0]-1,),
				acc[1] + (X[i, 0, -1]/max_value[1],),
			),
			range(X.shape[0]),
			(tuple(),tuple(),)
		)

		del X

		plt.plot(\
			x_list,
			y_list,
			'r+',
		)

		i = 0
		while i < len(x_list):
			plt.plot([x_list[i], x_list[i],], [0,1,])
			i+=20

		plt.show()

	if False:
		plot_queries_per_throughput_value()