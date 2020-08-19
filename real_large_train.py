from sklearn.svm import SVR
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import keras
from multiprocessing import Pool

def svr_train_main():
	X = json.load(open('first_week_normalized_data_set.json','rt'))

	train_indexes_list, valid_indexes_list = pickle.load(
		open('first_week_train_test_indexes_split.p','rb')
	)

	X_train = np.empty((len(train_indexes_list),len(X[0])))
	j=0
	for i in train_indexes_list:
		X_train[j] = X[i]
		# X_train[j,-1] *= 100
		# X_train[j,-1] = 1 - X_train[j,-1]
		j+=1

	X_valid = np.empty((len(valid_indexes_list),len(X[0])))
	j=0
	for i in valid_indexes_list:
		X_valid[j] = X[i]
		# X_valid[j,-1] *= 100
		# X_valid[j,-1] = 1 - X_valid[j,-1]
		j+=1

	if False:
		plt.plot(
			range(len(X)),
			tuple(map(lambda e: e[-1],X))
		)
		plt.show()
		exit(0)
	if False:
		plt.plot(
			range(X_train.shape[0]),
			X_train[:,-1]
		)
		plt.show()
		plt.plot(
			range(X_valid.shape[0]),
			X_valid[:,-1]
		)
		plt.show()
		exit(0)
	if False:
		indexes_lists_are_disjoint_flag = True
		for i in train_indexes_list:
			if i in valid_indexes_list:
				indexes_lists_are_disjoint_flag = False
				break
		if indexes_lists_are_disjoint_flag:
			print('Are disjoint !')
		else:
			print('Are not disjoint !')
		exit(0)

	del X

	# regr_1 = RandomForestRegressor(n_estimators=1600,n_jobs=-1)

	# Best for [0,1] normalization: (0.24393057095197093, 'poly 0.75 3 (0.24894810670624104, 0.0530996403342361, 0.24393057095197093, 0.05289685347900076)')
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

	if True:

		plt.plot(
			range(X_train.shape[0]),
			X_train[:,-1],
			'b-'
		)
		plt.plot(
			range(X_train.shape[0]),
			y_1,
			'r-'
		)

		plt.show()
		exit(0)

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

	mae = 0
	for i in range(X_valid.shape[0]):
		d = X_valid[i,-1] - y_1[i]

		if d < 0:
			d = -d

		mae += d

	print('mae: ' +  str(mae/X_valid.shape[0]))

	print()

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

	if False:
		X = json.load(open('first_week_normalized_data_set.json','rt'))
		train_indexes_list, valid_indexes_list = pickle.load(
			open('first_week_train_test_indexes_split.p','rb')
		)
	if False:
		X = json.load(open('first_week_normalized_data_set_trend.json','rt'))
		train_indexes_list, valid_indexes_list = pickle.load(
			open('first_week_train_test_indexes_split_trend.p','rb')
		)
	if True:
		X = json.load(open('first_week_normalized_data_set_site_thp.json','rt'))
		train_indexes_list, valid_indexes_list = pickle.load(
			open('first_week_train_test_indexes_split_site_thp.p','rb')
		)

	X_train = np.empty((len(train_indexes_list),len(X[0])))
	j=0
	for i in train_indexes_list:
		X_train[j] = X[i]
		# X_train[j,-1] = 1 - X_train[j,-1]
		j+=1

	X_valid = np.empty((len(valid_indexes_list),len(X[0])))
	j=0
	for i in valid_indexes_list:
		X_valid[j] = X[i]
		# X_valid[j,-1] = 1 - X_valid[j,-1]
		j+=1

	print(X_train.shape)
	print(X_valid.shape)

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

def get_model_1(ws, bins_no):
	inp_layer = keras.layers.Input(shape=(ws, bins_no,))

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=500)
	)(inp_layer)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=250,
			return_sequences=True,
		)
	)(x)
	# x = keras.layers.Dropout(0.05)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=250,)
	)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=100,)
	)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=25,)
	)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=1, activation='relu')
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_recurent_module(inp, bins_no):
	x = keras.layers.Reshape((bins_no,1))(inp)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=bins_no,
			return_sequences=False,
		)
	)(x)

	return x

def get_model_2_0(ws, bins_no):
	inp_layer = keras.layers.Input(shape=(ws, bins_no,))

	x = inp_layer

	# x = keras.layers.TimeDistributed(
	# 	keras.layers.Dense(units=500)
	# )(inp_layer)
	# x = keras.layers.LeakyReLU(alpha=0.3)(x)
	# x = keras.layers.Dropout(0.05)(x)

	a_num = 90

	# x = keras.layers.BatchNormalization()(x)
	# x = keras.layers.TimeDistributed(
	# 	keras.layers.Dense(units=a_num)
	# )(x)
	# x = keras.layers.LeakyReLU(alpha=0.3)(x)
	# x = keras.layers.Dropout(0.05)(x)

	# x = keras.layers.BatchNormalization()(x)

	x = keras.layers.Concatenate()([\
		get_recurent_module(\
			keras.layers.Lambda(lambda y: y[:,i,:], output_shape=(1,a_num))(x),\
			a_num\
		) for i in range(ws)\
	])
	x = keras.layers.Reshape((ws,2*a_num))(x)
	x = keras.layers.Dropout(0.05)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=a_num)
	)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.Dropout(0.05)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=a_num,
			return_sequences=True,
		)
	)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=a_num,)
	)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.Dropout(0.05)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=50)
	)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.Dropout(0.05)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=25)
	)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.Dropout(0.05)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=1, activation='relu')
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def nn_train_main_0():
	if False:
		X = json.load(open('first_week_normalized_data_set.json','rt'))
		train_indexes_list, valid_indexes_list = pickle.load(
			open('first_week_train_test_indexes_split.p','rb')
		)
	if False:
		X = json.load(open('first_week_normalized_data_set_site_thp.json','rt'))
		train_indexes_list, valid_indexes_list = pickle.load(
			open('first_week_train_test_indexes_split_site_thp.p','rb')
		)
	if False:
		X = json.load(open('first_week_normalized_data_set_100k_interp.json','rt'))
		train_indexes_list, valid_indexes_list = pickle.load(
			open('first_week_train_test_indexes_split_100k_interp.p','rb')
		)

	if False:
		X =\
		tuple(
			map(
				lambda e: e[500:],
				X
			)
		)

	if False:
		X = json.load(open('three_weeks_normalized_data_set.json','rt'))
		train_indexes_list, valid_indexes_list = pickle.load(
			open('three_weeks_train_test_indexes_split.p','rb')
		)
		out_folder = 'three_weeks_models/'

	if False:
		X = json.load(open('three_weeks_normalized_trend_data_set.json','rt'))
		train_indexes_list, valid_indexes_list = pickle.load(
			open('three_weeks_train_test_indexes_split_trend.p','rb')
		)
		out_folder = 'three_weeks_models_trend/'

	if False:
		window_size = 40

		train_indexes_list, valid_indexes_list =\
			list(filter(lambda e: e >= window_size - 1 , train_indexes_list)),\
			list(filter(lambda e: e >= window_size - 1 , valid_indexes_list))

		bin_size = len(X[0]) - 1

		X_train = np.empty((len(train_indexes_list),window_size,bin_size))
		y_train = np.empty((len(train_indexes_list),window_size,1))
		for i in range(len(train_indexes_list)):
			for c,j in zip(
					range(window_size),
					range(train_indexes_list[i] - window_size + 1, train_indexes_list[i] + 1),
				):
				X_train[i,c] = X[j][:-1]
				y_train[i,c] = X[j][-1]

		X_valid = np.empty((len(valid_indexes_list),window_size,bin_size))
		y_valid = np.empty((len(valid_indexes_list),window_size,1))
		for i in range(len(valid_indexes_list)):
			for c,j in zip(
					range(window_size),
					range(valid_indexes_list[i] - window_size + 1, valid_indexes_list[i] + 1)
				):
				X_valid[i,c] = X[j][:-1]
				y_valid[i,c] = X[j][-1]

		del X
		del valid_indexes_list
		del train_indexes_list

	if True:
		a,b,c,d = json.load( open( '/optane/mipopa/data_set_dumps/three_weeks_ready_to_train_90_bins_data_set.json' , 'rt') )
		X_train = np.array(a)
		y_train = np.array(b)
		X_valid = np.array(c)
		y_valid = np.array(d)
		out_folder = 'three_weeks_corr_2k_bins_thp/'


	if True:
		model = get_model_2_0(len(c[0]), len(c[0][0]))

		model.compile(
			optimizer=keras.optimizers.Adam(),
			loss='mean_absolute_percentage_error',
			metrics=['mae',]
		)
		del a
		del b
		del c
		del d
	if False:
		model = get_model_1(window_size, 1000)
		model.compile(
			optimizer=keras.optimizers.Adam(),
			loss='mean_absolute_percentage_error',
			metrics=['mae',]
		)

		old_model = keras.models.load_model(
			'/optane/mipopa/three_weeks_models_trend_corr/best_model.hdf5')

		old_model_idexes_list = []
		for i in range(len(old_model.layers)):
			if old_model.layers[i].name.startswith('time_distributed')\
				or old_model.layers[i].name.startswith('bidirectional'):
				old_model_idexes_list.append(i)

		j=0
		for i in range(len(model.layers)):
			if model.layers[i].name.startswith('time_distributed')\
				or model.layers[i].name.startswith('bidirectional'):
				model.layers[i].set_weights(old_model.layers[old_model_idexes_list[j]].get_weights())
				j+=1
		del old_model
		del old_model_idexes_list

	if False:
		model = get_model_2_0(window_size, 500)
		model.load_weights('models/model_weights.h5')
		model.compile(
				optimizer=keras.optimizers.Adam(),
				loss='mean_absolute_percentage_error',
				metrics=['mae',]
		)

	model.summary()

	print( X_train.shape )
	print( y_train.shape )
	print( X_valid.shape )
	print( y_valid.shape )

	model.fit(
		x=X_train,
		y=y_train,
		batch_size=32,
		epochs=100000,
		validation_data=(\
			X_valid,
			y_valid,
		),
		# callbacks=[
		# 	keras.callbacks.ModelCheckpoint(out_folder + "model_{epoch:04d}.hdf5", monitor='loss', save_best_only=True),
		# ]
	)

def get_model_3(ws, bins_no, out_len):
	inp_layer = keras.layers.Input(shape=(ws, bins_no,))

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=1000)
	)(inp_layer)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.Dropout(0.05)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=500,
			return_sequences=True,
		)
	)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=500)
	)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	# x = keras.layers.Dropout(0.05)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=250)
	)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	# x = keras.layers.Dropout(0.05)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=125)
	)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	# x = keras.layers.Dropout(0.05)(x)

	x = keras.layers.BatchNormalization()(x)
	x =	keras.layers.Dense(units=out_len, activation='relu')(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def nn_train_main_1():

	if True:
		X = json.load(open('three_weeks_normalized_data_set.json','rt'))
		train_indexes_list, valid_indexes_list = pickle.load(
			open('three_weeks_train_test_indexes_split.p','rb')
		)

	print( len(X[0][0]) )
	print( len(X[0][1]) )

	window_size = 30

	train_indexes_list, valid_indexes_list =\
		list(filter(lambda e: e >= window_size - 1 , train_indexes_list)),\
		list(filter(lambda e: e >= window_size - 1 , valid_indexes_list))

	bin_size = len(X[0][0])

	X_train = np.empty((len(train_indexes_list),window_size,bin_size))
	y_train = np.empty((len(train_indexes_list),window_size,len(X[0][1])))
	for i in range(len(train_indexes_list)):
		for c,j in zip(
				range(window_size),
				range(train_indexes_list[i] - window_size + 1, train_indexes_list[i] + 1),
			):
			X_train[i,c] = X[j][0]
			y_train[i,c] = X[j][1]

	X_valid = np.empty((len(valid_indexes_list),window_size,bin_size))
	y_valid = np.empty((len(valid_indexes_list),window_size,len(X[0][1])))
	for i in range(len(valid_indexes_list)):
		for c,j in zip(
				range(window_size),
				range(valid_indexes_list[i] - window_size + 1, valid_indexes_list[i] + 1),
			):
			X_valid[i,c] = X[j][0]
			y_valid[i,c] = X[j][1]

	model = get_model_3(window_size, bin_size, len(X[0][1]))

	del X

	model.compile(
			optimizer=keras.optimizers.Adam(),
			loss='mean_absolute_percentage_error',
			metrics=['mae',]
	)

	model.summary()

	model.fit(
		x=X_train,
		y=y_train,
		batch_size=32,
		epochs=300,
		validation_data=(\
			X_valid,
			y_valid,
		),
	)

def get_model_4(ws, bins_no, out_len):

	def get_module(i_l):
		x = keras.layers.BatchNormalization()(i_l)
		x = keras.layers.Bidirectional(
			keras.layers.LSTM(
				units=125,
				return_sequences=True,
			)
		)(x)

		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.TimeDistributed(
			keras.layers.Dense(units=125)
		)(x)
		x = keras.layers.LeakyReLU(alpha=0.3)(x)

		x = keras.layers.BatchNormalization()(x)
		return	keras.layers.Dense(units=1, activation='relu')(x)

	inp_layer = keras.layers.Input(shape=(ws, bins_no,))

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=3000)
	)(inp_layer)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.Dropout(0.05)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=1000)
	)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)
	x = keras.layers.Dropout(0.05)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=500,
			return_sequences=True,
		)
	)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=500)
	)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=250)
	)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)

	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=125)
	)(x)
	x = keras.layers.LeakyReLU(alpha=0.3)(x)

	return keras.models.Model(
		inputs=inp_layer,
		outputs=[ get_module(x) for _ in range(out_len) ]
	)

def nn_train_main_2():

	if True:
		X = json.load(open('first_week_normalized_data_set_cern_all_clients.json','rt'))
		train_indexes_list, valid_indexes_list = pickle.load(
			open('first_week_train_test_indexes_split_cern_all_clients.p','rb')
		)

	print( len(X[0][0]) )
	print( len(X[0][1]) )

	window_size = 30

	train_indexes_list, valid_indexes_list =\
		list(filter(lambda e: e >= window_size - 1 , train_indexes_list)),\
		list(filter(lambda e: e >= window_size - 1 , valid_indexes_list))

	bin_size = len(X[0][0])

	X_train = np.empty((len(train_indexes_list),window_size,bin_size))
	y_train_list = [np.empty((len(train_indexes_list),window_size,1)) for _ in X[0][1]]
	for i in range(len(train_indexes_list)):
		for c,j in zip(
				range(window_size),
				range(train_indexes_list[i] - window_size + 1, train_indexes_list[i] + 1),
			):
			X_train[i,c] = X[j][0]
			for l in range(len(X[0][1])):
				y_train_list[l][i,c]=X[j][1][l]

	X_valid = np.empty((len(valid_indexes_list),window_size,bin_size))
	y_valid_list = [np.empty((len(valid_indexes_list),window_size,1)) for _ in X[0][1]]
	for i in range(len(valid_indexes_list)):
		for c,j in zip(
				range(window_size),
				range(valid_indexes_list[i] - window_size + 1, valid_indexes_list[i] + 1),
			):
			X_valid[i,c] = X[j][0]
			for l in range(len(X[0][1])):
				y_valid_list[l][i,c]=X[j][1][l]

	model = get_model_4(window_size, bin_size, len(X[0][1]))

	del X

	model.compile(
			optimizer=keras.optimizers.Adam(),
			loss='mean_absolute_percentage_error',
			metrics=['mae',]
	)

	model.summary()

	model.fit(
		x=X_train,
		y=y_train_list,
		batch_size=32,
		epochs=300,
		validation_data=(\
			X_valid,
			y_valid_list,
		),
	)

def dump_predictions_main():

	model = keras.models.load_model('three_weeks_models_trend/model_0159.hdf5')

	if True:
		X = json.load(open('data_set_dumps/three_weeks_normalized_trend_data_set.json','rt'))
		train_indexes_list, valid_indexes_list = pickle.load(
			open('data_set_dumps/three_weeks_train_test_indexes_split_trend.p','rb')
		)

	window_size = 40

	train_indexes_list, valid_indexes_list =\
		list(filter(lambda e: e >= window_size - 1 , train_indexes_list)),\
		list(filter(lambda e: e >= window_size - 1 , valid_indexes_list))

	bin_size = len(X[0]) - 1

	X_train = np.empty((len(train_indexes_list),window_size,bin_size))
	y_train = np.empty((len(train_indexes_list),window_size,1))
	for i in range(len(train_indexes_list)):
		for c,j in zip(
				range(window_size),
				range(train_indexes_list[i] - window_size + 1, train_indexes_list[i] + 1),
			):
			X_train[i,c] = X[j][:-1]
			y_train[i,c] = X[j][-1]

	X_valid = np.empty((len(valid_indexes_list),window_size,bin_size))
	y_valid = np.empty((len(valid_indexes_list),window_size,1))
	for i in range(len(valid_indexes_list)):
		for c,j in zip(
				range(window_size),
				range(valid_indexes_list[i] - window_size + 1, valid_indexes_list[i] + 1)
			):
			X_valid[i,c] = X[j][:-1]
			y_valid[i,c] = X[j][-1]

	y_pred = model.predict( X_train )

	ground_truth_list = list()
	predicted_list = list()

	for i in range(y_pred.shape[0]):
		ground_truth_list.append(
			(
				train_indexes_list[i],
				y_train[i,0,0],
			)
		)
		predicted_list.append(
			(
				train_indexes_list[i],
				y_pred[i,0,0],
			)
		)
	y_pred_train = y_pred

	y_pred = model.predict( X_valid )
	for i in range(y_pred.shape[0]):
		ground_truth_list.append(
			(
				valid_indexes_list[i],
				y_valid[i,0,0],
			)
		)
		predicted_list.append(
			(
				valid_indexes_list[i],
				y_pred[i,0,0],
			)
		)

	if False:
		if window_size - 1 in train_indexes_list:
			ind = train_indexes_list.index(window_size - 1)

			for i in range(window_size-1):
				ground_truth_list.append(
					(
						i,
						y_train[ind,i,0],
					)
				)
				predicted_list.append(
					(
						i,
						y_pred_train[ind,i,0],
					)
				)
		else:
			ind = valid_indexes_list.index(window_size - 1)

			for i in range(window_size-1):
				ground_truth_list.append(
					(
						i,
						y_valid[ind,i,0],
					)
				)
				predicted_list.append(
					(
						i,
						y_pred[ind,i,0],
					)
				)

	pickle.dump(
		(
			tuple(
				map(
					lambda e: e[1],
					sorted(ground_truth_list)
				)
			),
			tuple(
				map(
					lambda e: e[1],
					sorted(predicted_list)
				)
			),
		),
		open(
			'network_results.p',
			'wb'
		)
	)

def dump_predictions_main_1():
	a,b,_,_ = json.load( open( '/optane/mipopa/data_set_dumps/three_weeks_ready_to_train_trend_data_set.json' , 'rt') )
	X_train = np.array(a)
	y_train = np.array(b)

	del a
	del b

	y_pred = keras.models.load_model('three_weeks_models_trend_corr/model_0082.hdf5').predict( X_train )

	ground_truth_list = list()
	predicted_list = list()

	for i in range(y_pred.shape[0]):
		ground_truth_list.append(y_train[i,0,0])
		predicted_list.append(y_pred[i,0,0])

	pickle.dump(
		(
			ground_truth_list,
			predicted_list,
		),
		open(
			'network_results.p',
			'wb'
		)
	)

if __name__ == '__main__':
	global n_proc

	n_proc = 95

	dump_predictions_main_1()