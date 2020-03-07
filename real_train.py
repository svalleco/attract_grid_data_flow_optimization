import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
from statistics import mean
import matplotlib.pyplot as plt
# import keras
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from multiprocessing import Pool

def get_model():
	model = keras.models.Sequential()

	model.add(
		keras.layers.Dense(
			units=32,
			activation='relu',
			input_shape=(192448,)
		)
	)

	model.add(
		keras.layers.Dense(units=16, activation='relu')
	)

	model.add(
		keras.layers.Dense(units=8, activation='relu')
	)

	model.add(
		keras.layers.Dense(units=4, activation='relu')
	)

	model.add(
		keras.layers.Dense(units=2, activation='relu')
	)

	model.add(
		keras.layers.Dense(units=1, activation='relu')
	)

	return model

def generate_X_0(queries_dict=pickle.load(open('first_option_cern.p','rb'))):
	X = np.zeros((len(queries_dict.keys()),2*96224+1))

	i = 0
	for k in queries_dict.keys():

		j = 0

		for q_tuple in sorted(queries_dict[k]):

			X[i,j] = q_tuple[0]
			X[i,j+1] = q_tuple[-1]

			j+=2

		X[i,-1] = k[1]

		i+=1

	return X

def generate_X_1(queries_dict=pickle.load(open('first_option_cern.p','rb'))):
	X = np.zeros((len(queries_dict.keys()),96224+1))

	i = 0
	for k in queries_dict.keys():

		j = 0

		for q_tuple in sorted(queries_dict[k]):

			X[i,j+1] = q_tuple[-1]

			j+=1

		X[i,-1] = k[1]

		i+=1

	return X

def analyse_0():
	a_list = tuple()

	b_list = tuple()

	min_v, max_v = 9576450800000, -1

	prv = -2

	for k,v in sorted(pickle.load(open('first_option_cern.p','rb')).items()):

		b_list += (k[1],)

		a = 0

		for q_tuple in v:

			a += q_tuple[-1]

		a_list += (a,)

		if a_list[-1] > max_v: max_v = a_list[-1]
		if a_list[-1] < min_v: min_v = a_list[-1]

		if prv > k[0]:
			print("mai mare !")
		prv = k[0]

	a_list = tuple(
		map(
			lambda e: 1 - (e - min_v) / (max_v-min_v),
			a_list
		)
	)

	plt.plot(
		range(len(a_list)),
		a_list,
	)

	plt.plot(
		range(len(b_list)),
		b_list,
	)

	plt.show()

def train_main():

	X = generate_X_1()

	print(X.shape)

	if False:
		model = get_model()

		model.compile(
			optimizer=keras.optimizers.Adam(),
			loss='mean_absolute_percentage_error',
			metrics=['mae','mse']
		)

		valid_indexes_list = random.sample(
			range(X.shape[0]),
			round(0.2 * X.shape[0])
		)

		train_indexes_list = tuple(filter(lambda e: e not in valid_indexes_list,range(X.shape[0])))

		model.summary()

		X_valid = np.empty((len(valid_indexes_list), X.shape[1]))

		for i in range(len(valid_indexes_list)):
			X_valid[i] = X[valid_indexes_list[i]]

		X_train = np.empty((len(train_indexes_list), X.shape[1]))

		for i in range(len(train_indexes_list)):
			X_train[i] = X[train_indexes_list[i]]

		del X

		model.fit(
			x=X_train[:,:-1],
			y=np.expand_dims(X_train[:,-1],axis=-1),
			epochs=200,
			validation_data=(\
				X_valid[:,:-1],
				np.expand_dims(X_valid[:,-1],axis=-1),
			)
		)

	plt.plot(
		range(X.shape[0]),
		np.sum(X[:,:-1], -1)
	)

	plt.plot(
		range(X.shape[0]),
		X[:,-1]
	)

	plt.show()

def analyse_1():

	queries_dict = pickle.load(open('first_option_cern.p','rb'))

	print(f'Number of thp values: {len(tuple(queries_dict.keys()))}')

	print(f'throughput time: {min(queries_dict.keys(), key=lambda p: p[0])[0]} {max(queries_dict.keys(), key=lambda p: p[0])[0]}')

	print(f'throughput value: {min(queries_dict.keys(), key=lambda p: p[1])[1]} {max(queries_dict.keys(), key=lambda p: p[1])[1]}')

	a = tuple(map(lambda k: len(queries_dict[k]) , queries_dict.keys()))

	print(f'max q per thp: {max(a)}')
	print(f'avg q per thp: {mean(a)}')
	print(f'min q per thp: {min(a)}')

	max_q_per_time_moment = -1
	min_q_per_time_moment = 9576450800000

	min_tm_per_thp, max_tm_per_thp = 9576450800000, -1

	q_count = 0

	for k in queries_dict.keys():
		d = dict()
		tm_set = set()

		tm_count = 0

		for q_list in queries_dict[k]:

			if q_list[0] in d:
				d[q_list[0]] += 1
			else:
				d[q_list[0]] = 1

			if q_list[0] not in tm_set:
				tm_set.add( q_list[0] )
				tm_count += 1

		q_count += len(queries_dict[k])

		if tm_count > max_tm_per_thp: max_tm_per_thp = tm_count
		if tm_count < min_tm_per_thp: min_tm_per_thp = tm_count

		for kk in d.keys():
			if d[kk] > max_q_per_time_moment:
				max_q_per_time_moment = d[kk]
			if d[kk] < min_q_per_time_moment:
				min_q_per_time_moment = d[kk]

	print(f'min q per time moment: {min_q_per_time_moment}')
	print(f'max q per time moment: {max_q_per_time_moment}')
	print(f'min tm per thp: {min_tm_per_thp}')
	print(f'max tm per thp: {max_tm_per_thp}')
	print(f'total count of queries: {q_count}')

	thp_list = sorted(queries_dict.keys())

	min_spacing, max_spacing = 9576450800000, -1
	a = 0
	for i in range(1,len(thp_list)):
		diff = thp_list[i][0] - thp_list[i-1][0]
		a += diff
		if diff < min_spacing: min_spacing = diff
		if diff > max_spacing: max_spacing = diff

	print(f'max thp spacing in time: {max_spacing}')
	print(f'avg thp spacing in time: {a/(len(thp_list)-1)}')
	print(f'min thp spacing in time: {min_spacing}')

def generate_X_2(X, indexes_list, window_size, batch_size):

	while True:
		sub_X = np.empty((\
			batch_size,\
			window_size,\
			X.shape[-1],\
		))

		i = 0
		for index in random.sample(indexes_list,batch_size):
			sub_X[i] = X[index:index+window_size]
			i+=1

		yield sub_X[:,:,:-1],\
			np.expand_dims( sub_X[:,:,-1] , axis=-1 )

def train_main_1(window_size):
	# queries_dict, cl_dict, se_dict = pickle.load(open('queries_dict_and_mapping.p','rb'))

	X = pickle.load(open('normalized_binned_thp_queries_array.p','rb'))

	valid_indexes_list = random.sample(range(X.shape[0]-window_size),round(0.2*(X.shape[0]-window_size)))

	train_indexes_list = tuple(filter(lambda e: e not in valid_indexes_list, range(X.shape[0]-window_size)))

	model = keras.models.Sequential()

	model.add(
		keras.layers.Bidirectional(
			keras.layers.LSTM(units=100,
				return_sequences=True,
			),
			input_shape=(window_size, 200),
		)
	)

	model.add(
		keras.layers.TimeDistributed(
			keras.layers.Dense(units=100, activation='relu')
		)
	)


	model.add(
		keras.layers.TimeDistributed(
			keras.layers.Dense(units=20, activation='relu')
		)
	)


	model.add(
		keras.layers.TimeDistributed(
			keras.layers.Dense(units=1, activation='relu')
		)
	)

	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	model.summary()

	model.fit_generator(
		generator=generate_X_2(X,train_indexes_list,window_size,32),
		validation_data=generate_X_2(X,valid_indexes_list,window_size,32),
		steps_per_epoch=100,
		validation_steps=20,
		epochs=300,
	)

def analyse_2():
	queries_dict, cl_dict, se_dict = pickle.load(open('queries_dict_and_mapping.p','rb'))

	print('max client index: ' + str( max(map(lambda k: cl_dict[k], cl_dict.keys())) ))

	print('max se index: ' + str( max(map(lambda k: se_dict[k], se_dict.keys())) ))

	max_viable_se_count = -1

	for k in queries_dict.keys():
		for q_line in queries_dict[k]:
			if len(q_line[1]) > max_viable_se_count:
				max_viable_se_count = len(q_line[1])

	print('max number of ses: ' + str(max_viable_se_count))

def get_model_2_1(a,b):
	model = keras.models.Sequential()

	model.add(
		keras.layers.TimeDistributed(
			keras.layers.Dense(units=100, activation='relu'),
			input_shape=(a, b)
		)
	)

	model.add(
		keras.layers.Bidirectional(
			keras.layers.LSTM(
				units=50,
				return_sequences=True,
			)
		)
	)

	model.add(
		keras.layers.TimeDistributed(
			keras.layers.Dense(units=100, activation='relu')
		)
	)

	model.add(
		keras.layers.Bidirectional(
			keras.layers.LSTM(
				units=50,
				return_sequences=True,
			)
		)
	)

	model.add(
		keras.layers.TimeDistributed(
			keras.layers.Dense(units=1, activation='relu')
		)
	)

	return model

def get_recurent_module(inp, bins_no):
	x = keras.layers.Reshape((bins_no,1))(inp)
	# x = keras.layers.Permute((0,2,1,))(inp)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=bins_no,
			return_sequences=False,
		)
	)(x)

	return x

def get_model_2_0(ws, bins_no):
	inp_layer = keras.layers.Input(shape=(ws, bins_no,))

	x = keras.layers.Concatenate()([\
		get_recurent_module(\
			keras.layers.Lambda(lambda x: x[:,i,:], output_shape=(1,bins_no))(inp_layer),\
			bins_no\
		) for i in range(ws)\
	])

	x = keras.layers.Reshape((ws,2*bins_no))(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=100, activation='tanh')
	)(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=100,
			return_sequences=True,
		)
	)(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=100,)
	)(x)

	x = keras.layers.LeakyReLU(alpha=0.3)(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=50)
	)(x)

	x = keras.layers.LeakyReLU(alpha=0.3)(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=25)
	)(x)

	x = keras.layers.LeakyReLU(alpha=0.3)(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=1, activation='relu')
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def train_main_2(window_size):

	X = pickle.load(open('normalized_binned_thp_queries_array.p','rb'))

	if False:
		print(X.shape)

		a_list = []
		b_list = []

		for i in range(X.shape[0]):

			a_list.append(sum(X[i,:100]))

			b_list.append(X[i,100])

		plt.plot(range(X.shape[0]),
			list(
				map(lambda e: (e - min(a_list)) / (max(a_list)-min(a_list)),
				a_list)
			))

		plt.plot(range(X.shape[0]), b_list)

		plt.savefig('train_main_test.png')

		plt.show()

		exit(0)

	if False:

		valid_indexes_list = random.sample(range(X.shape[0]-window_size),round(0.2*(X.shape[0]-window_size)))

		train_indexes_list = tuple(filter(lambda e: e not in valid_indexes_list, range(X.shape[0]-window_size)))

	if True:
		train_indexes_list, valid_indexes_list = pickle.load(
			open('train_test_indexes_split.p','rb')
		)

	model = get_model_2_0(window_size, X.shape[-1] - 1)

	model.compile(
		optimizer=keras.optimizers.RMSprop(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	model.summary()

	model.fit_generator(
		generator=generate_X_2(X,train_indexes_list,window_size,16),
		validation_data=generate_X_2(X,valid_indexes_list,window_size,16),
		steps_per_epoch=10,
		validation_steps=2,
		epochs=300,
	)

def generate_X_3(X, indexes_list, batch_size):

	while True:
		sub_X = np.empty((batch_size,X.shape[-1],))

		i = 0
		for index in random.sample(indexes_list,batch_size):
			sub_X[i] = X[index]
			i+=1

		yield sub_X[:,:-1],\
			np.expand_dims( sub_X[:,-1] , axis=-1 )

def get_model_3(bins_no=200):
	model = keras.models.Sequential()

	model.add(keras.layers.Dense(units=100, input_shape=(bins_no,)))

	model.add(keras.layers.LeakyReLU(alpha=0.3))

	model.add(keras.layers.Dense(units=50,))

	model.add(keras.layers.LeakyReLU(alpha=0.3))

	model.add(keras.layers.Dense(units=25,))

	model.add(keras.layers.LeakyReLU(alpha=0.3))

	model.add(keras.layers.Dense(units=1,activation='relu'))

	return model

def train_main_3():
	X = pickle.load(open('normalized_binned_thp_queries_array.p','rb'))

	if False:
		new_X = np.empty(
			(
				len(tuple(filter(lambda i: X[i,-1] < 0.6,range(X.shape[0])))),
				X.shape[-1]
			)
		)

		j = 0
		for line in X:
			if line[-1] < 0.6:
				new_X[j] = line
				j+=1

		X = new_X

	print(X.shape)

	if False:

		valid_indexes_list = random.sample(range(X.shape[0]-window_size),round(0.2*(X.shape[0]-window_size)))

		train_indexes_list = tuple(filter(lambda e: e not in valid_indexes_list, range(X.shape[0]-window_size)))

	if True:
		train_indexes_list, valid_indexes_list = pickle.load(
			open('train_test_indexes_split.p','rb')
		)

	model = get_model_3()

	model.compile(
		optimizer=keras.optimizers.Nadam(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	model.summary()

	model.fit_generator(
		generator=generate_X_3(X,train_indexes_list,24),
		validation_data=generate_X_3(X,valid_indexes_list,24),
		steps_per_epoch=100,
		validation_steps=20,
		epochs=300,
	)

def tree_train_main():
	X = pickle.load(open('normalized_binned_thp_queries_array.p','rb'))

	train_indexes_list, valid_indexes_list = pickle.load(
		open('train_test_indexes_split.p','rb')
	)

	X_train = np.empty((len(train_indexes_list),X.shape[-1]))
	j=0
	for i in train_indexes_list:
		X_train[j] = X[i]
		j+=1

	X_valid = np.empty((len(valid_indexes_list),X.shape[-1]))
	j=0
	for i in valid_indexes_list:
		X_valid[j] = X[i]
		j+=1


	# regr_1 = RandomForestRegressor(n_estimators=1600,n_jobs=-1)
	regr_1 = SVR(
		kernel='poly',
		C=10,
		degree=2,
		verbose=True,
	)

	print('Will start training the tree !')

	regr_1.fit(X_train[:,:-1], X_train[:,-1])

	y_1 = regr_1.predict(X_train[:,:-1])

	print()

	mape = 0
	div = 0
	for i in range(X_train.shape[0]):
		d = X_train[i,-1] - y_1[i]

		if d < 0:
			d = -d

		# mape += d / max((X_train[i,-1],1,))
		if X_train[i,-1] != 0:
			mape += d/X_train[i,-1]
			div += 1

	print('mape: ' +  str(mape/div))
	print('\tdiv: ' + str(div) + '/' + str(X_train.shape[0]))

	mae = 0
	for i in range(X_train.shape[0]):
		d = X_train[i,-1] - y_1[i]

		if d < 0:
			d = -d

		mae += d

	print('mae: ' +  str(mae/X_train.shape[0]))

	print()

	y_1 = regr_1.predict(X_valid[:,:-1])

	mape = 0
	div = 0
	for i in range(X_valid.shape[0]):
		d = X_valid[i,-1] - y_1[i]

		if d < 0:
			d = -d

		# mape += d / max((X_valid[i,-1],1,))
		if X_valid[i,-1] != 0:
			mape += d/X_valid[i,-1]
			div += 1

	print('mape: ' +  str(mape/div))
	print('\tdiv: ' + str(div) + '/' + str(X_valid.shape[0]))

def train_per_proc(i):
	if len(g_indexes_pair[i]) == 3:
		regr_1 = SVR(
			kernel='poly',
			C=g_c_list[g_indexes_pair[i][1]],
			degree=g_degree_list[g_indexes_pair[i][2]],
		)
	else:
		regr_1 = SVR(
			kernel=g_kernel_list[g_indexes_pair[i][0]],
			C=g_c_list[g_indexes_pair[i][1]],
		)

	regr_1.fit(X_train[:,:-1], X_train[:,-1])

	y_1 = regr_1.predict(X_train[:,:-1])

	mape = 0
	div = 0
	for i in range(X_train.shape[0]):
		d = X_train[i,-1] - y_1[i]

		if d < 0:
			d = -d

		if X_train[i,-1] != 0:
			mape += d/X_train[i,-1]
			div += 1
	train_mape = mape / div

	mae = 0
	for i in range(X_train.shape[0]):
		d = X_train[i,-1] - y_1[i]

		if d < 0:
			d = -d

		mae += d
	train_mae = mae / X_train.shape[0]

	y_1 = regr_1.predict(X_valid[:,:-1])

	mape = 0
	div = 0
	for i in range(X_valid.shape[0]):
		d = X_valid[i,-1] - y_1[i]

		if d < 0:
			d = -d

		if X_valid[i,-1] != 0:
			mape += d/X_valid[i,-1]
			div += 1
	valid_mape = mape / div

	mae = 0
	for i in range(X_valid.shape[0]):
		d = X_valid[i,-1] - y_1[i]

		if d < 0:
			d = -d

		mae += d
	valid_mae = mae / X_valid.shape[0]

	return (train_mape, train_mae, valid_mape, valid_mae,)

def my_grid_search_main():
	global g_kernel_list, g_c_list, g_indexes_pair, X_train, X_valid,\
		g_degree_list

	g_kernel_list = ['poly','rbf','sigmoid',]
	g_c_list = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 10,]
	g_degree_list = [2,3,4,5,6]

	X = pickle.load(open('normalized_binned_thp_queries_array.p','rb'))

	train_indexes_list, valid_indexes_list = pickle.load(
		open('train_test_indexes_split.p','rb')
	)

	X_train = np.empty((len(train_indexes_list),X.shape[-1]))
	j=0
	for i in train_indexes_list:
		X_train[j] = X[i]
		j+=1

	X_valid = np.empty((len(valid_indexes_list),X.shape[-1]))
	j=0
	for i in valid_indexes_list:
		X_valid[j] = X[i]
		j+=1

	g_indexes_pair = []
	for i in range(len(g_kernel_list)):
		for j in range(len(g_c_list)):
			if g_kernel_list[i] != 'poly':
				g_indexes_pair.append((i,j,))
			else:
				for k in range(len(g_degree_list)):
					g_indexes_pair.append((i,j,k,))

	best_tuple = (1, None,)

	i = 0
	for r_tuple in Pool(n_proc).map(train_per_proc,range(len(g_indexes_pair))):
		if len(g_indexes_pair[i]) != 3:
			s = g_kernel_list[g_indexes_pair[i][0]] + ' ' + str(g_c_list[g_indexes_pair[i][1]]) + ' ' + str(r_tuple)
		else:
			s = g_kernel_list[g_indexes_pair[i][0]] + ' ' + str(g_c_list[g_indexes_pair[i][1]]) + ' ' + str(g_degree_list[g_indexes_pair[i][2]]) + ' ' + str(r_tuple)

		print(s)

		if r_tuple[2] < best_tuple[0]:
			best_tuple = (r_tuple[2],s)

		i+=1

	print()
	print(best_tuple)

def plot_regressors_results_main():
	if False:
		X = pickle.load(open('normalized_binned_thp_queries_array.p','rb'))

		regr_1 = SVR(
			kernel='poly',
			C=10,
			degree=2,
		)

		train_indexes_list, valid_indexes_list = pickle.load(
			open('train_test_indexes_split.p','rb')
		)

		X_train = np.empty((len(train_indexes_list),X.shape[-1]))
		j=0
		for i in train_indexes_list:
			X_train[j] = X[i]
			j+=1

		X_valid = np.empty((len(valid_indexes_list),X.shape[-1]))
		j=0
		for i in sorted(valid_indexes_list):
			X_valid[j] = X[i]
			j+=1

		regr_1.fit(X_train[:,:-1], X_train[:,-1])

		pickle.dump(
			(
				tuple(range(X_train.shape[0])),
				X_train[:,-1],
				tuple(range(X_train.shape[0])),
				regr_1.predict(X_train[:,:-1]),
				tuple(range(X_valid.shape[0])),
				X_valid[:,-1],
				tuple(range(X_valid.shape[0])),
				regr_1.predict(X_valid[:,:-1]),
			),
			open(
				'pipe_2.p',
				'wb'
			)
		)
	else:
		a,b,c,d,e,f,g,h = pickle.load(open('pipe_2.p','rb'))

		# plt.figure()
		plt.subplot(211)
		plt.plot(a,b,label='ground_truth')
		plt.plot(c,d,label='prediction')

		plt.xlabel('Index In The Train set')
		plt.ylabel('Normalized Throughput Value')
		plt.legend()

		plt.subplot(212)
		plt.plot(e,f,label='ground_truth')
		plt.plot(g,h,label='prediction')

		plt.xlabel('Index In The Validation set')
		plt.ylabel('Normalized Throughput Value')
		plt.legend()

		plt.show()


if __name__ == '__main__':
	global n_proc
	n_proc = 40

	if False:
		train_main()

	if False:
		analyse_1()

	if False:
		train_main_3()

	if False:
		train_main_2(20)

	if False:
		my_grid_search_main()

	if False:
		tree_train_main()

	if False:
		plot_regressors_results_main()