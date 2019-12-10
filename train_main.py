import numpy as np
import pickle
from keras.models import Sequential, Model
import keras.layers
import keras.metrics
import keras.optimizers
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
import random
from csv import reader
from os import listdir
import random
import matplotlib.pyplot as plt

def get_model_1(time_window_size, optimizer_obj):
	model = Sequential()

	model.add(
		keras.layers.TimeDistributed(
			keras.layers.Dense(units=160, activation='relu'),
			input_shape=(time_window_size, 243)
		)
	)

	# model.add(
	# 	keras.layers.Dropout(0.1)
	# )

	model.add(
		keras.layers.TimeDistributed(
			keras.layers.Dense(units=80, activation='relu'),
		)
	)

	model.add(
		keras.layers.Dropout(0.1)
	)

	model.add(
		keras.layers.TimeDistributed(
			keras.layers.Dense(units=40, activation='relu'),
		)
	)

	model.add(
		keras.layers.Dropout(0.1)
	)

	model.add(
		keras.layers.Bidirectional(
			keras.layers.LSTM(units=10, return_sequences=True)
		)
	)

	# model.add(
	# 	keras.layers.LSTM(units=10, return_sequences=True)
	# 	)

	model.add(
		keras.layers.Dropout(0.1)
	)

	model.add(
		keras.layers.TimeDistributed(
			keras.layers.Dense(units=10, activation='relu')
		)
	)

	model.add(
		keras.layers.Dropout(0.1)
	)

	model.add(
		keras.layers.TimeDistributed(
			keras.layers.Dense(units=1, activation='relu')
		)
	)

	model.compile(
		optimizer=optimizer_obj,
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	return model

def get_model_2(window_size):
	x0 = keras.layers.Input(shape=((window_size, 242)))

	x = keras.layers.TimeDistributed(
			keras.layers.Dense(units=80, activation='relu'),
		)(x0)

	# x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.TimeDistributed(
			keras.layers.Dense(units=20, activation='relu'),
		)(x)

	# x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.TimeDistributed(
			keras.layers.Dense(units=1, activation='relu'),
		)(x)

	x1 = keras.layers.Input(shape=(window_size,1))

	x = keras.layers.concatenate([x1, x])

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(units=4, return_sequences=True)
	)(x)

	# x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.TimeDistributed(
			keras.layers.Dense(units=4, activation='relu'),
	)(x)

	x = keras.layers.TimeDistributed(
			keras.layers.Dense(units=1, activation='relu'),
		)(x)

	model = Model(inputs=[x0,x1], outputs=x)

	model.compile(
		optimizer=keras.optimizers.Nadam(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	return model

def normalize0(X):
	max_value = -np.ones(3)

	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			max_value = np.maximum(max_value, X[i,j,-3:])

	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			X[i,j,-3:-1] = 2 * X[i,j,-3:-1] / max_value[:-1] - 1
			X[i,j,-1] = X[i,j,-1] / max_value[-1]

	return X

def normalize1(X):
	max_value = -np.ones(3)
	for i in range(X.shape[0]):
		max_value = np.maximum(max_value, X[i,-3:])

	for i in range(X.shape[0]):
		X[i,-3:-1] = 2 * X[i,-3:-1] / max_value[:-1] - 1
		X[i,-1] = X[i,-1] / max_value[-1]

	return X

def shuffle(X):
	l = list(range(X.shape[0]))

	random.shuffle(l)

	new_X = np.empty(X.shape)

	for i,j in enumerate(l):
		new_X[i] = X[j]

	return new_X

def transform_to_time_window(X, window_size):
	new_X = np.empty((X.shape[0]-window_size,window_size,X.shape[-1],))
	for i in range(new_X.shape[0]):
		new_X[i] = X[i:i+window_size]
	return new_X

def generate0(X, index_list, batch_size, window_size):
	while True:
		sub_X = np.empty((\
			batch_size,\
			window_size,\
			X.shape[-1],\
		))

		i = 0
		for index in random.sample(index_list,batch_size):
			sub_X[i] = X[index:index+window_size]
			i+=1

		yield sub_X[:,:,:-1],\
			np.expand_dims( sub_X[:,:,-1] , axis=-1 )

def generate1(X, index_list, batch_size, window_size):
	while True:
		sub_X = np.empty((\
			batch_size,\
			window_size,\
			X.shape[-1],\
		))

		i = 0
		for index in random.sample(index_list,batch_size):
			sub_X[i] = X[index:index+window_size]
			i+=1

		yield [sub_X[:,:,:-2],sub_X[:,:,-2:-1],],\
			np.expand_dims( sub_X[:,:,-1] , axis=-1 )

def dump_train_and_validation_indexes0(\
	val_percentage,
	window_size,
	pipe_name='./pipe.p'):

	train_index_list = list(range(\
		pickle.load(open(pipe_name,'rb')).shape[0]-window_size
	))

	val_index_list = []

	val_size = round(\
		val_percentage * len(train_index_list)\
	)

	while len(val_index_list) != val_size:
		val_index_list.append(random.choice(train_index_list))
		train_index_list.remove(val_index_list[-1])

	pickle.dump(\
		(train_index_list,val_index_list,),
		open('train_validation_indexes_lists.p', 'wb')
	)

def dump_train_and_validation_indexes1(\
	val_percentage,
	window_size,
	pipe_name='./pipe.p'):

	train_index_list = list(\
		filter(
			lambda ind: ind % window_size == 0,
			range(\
				pickle.load(open(pipe_name,'rb')).shape[0]-window_size
			)
		)
	)

	val_index_list = []

	val_size = round(\
		val_percentage * len(train_index_list)\
	)

	while len(val_index_list) != val_size:
		val_index_list.append(random.choice(train_index_list))
		train_index_list.remove(val_index_list[-1])

	pickle.dump(\
		(train_index_list,val_index_list,),
		open('train_validation_indexes_lists.p', 'wb')
	)

def train_main_0(\
	optimizer,
	dump_csv_name,
	window_size,
	):
	X = normalize1(pickle.load(open('./pipe.p','rb')))

	train_index_list, val_index_list = pickle.load(\
		open('train_validation_indexes_lists.p' , 'rb' )
	)

	model = get_model_1(\
		window_size,
		optimizer,
	)

	model.summary()

	model.fit_generator(
		generator=generate0(X,train_index_list,4,window_size),
		validation_data=generate0(X,val_index_list,4,window_size),
		steps_per_epoch=1000,
		validation_steps=200,
		epochs=300,
		callbacks=[\
			TensorBoard(log_dir='./board/'),
			# ModelCheckpoint("models/model_{epoch:04d}.hdf5", monitor='loss', period=10),
			# CSVLogger(dump_csv_name, append=True, separator=';'),
		]
	)

def train_main_1():
	X = normalize1(pickle.load(open('./pipe.p','rb')))

	window_size = 40

	train_index_list, val_index_list = pickle.load(\
		open('train_validation_indexes_lists.p' , 'rb' )
	)

	model = get_model_2(window_size)

	model.summary()

	model.fit_generator(
		generator=generate1(X,train_index_list,64,window_size),
		validation_data=generate1(X,val_index_list,64,window_size),
		steps_per_epoch=1000,
		validation_steps=200,
		epochs=2000,
		callbacks=[TensorBoard(log_dir='./board/')]
	)

def test_main():
	X = normalize1(pickle.load( open( './test_pipeline_set.p' , 'rb' ) ))

	window_size = 20

	model = get_model_1()

	model.load_weights('models/model_0091.hdf5')

	results = model.evaluate_generator(\
		generate0(X, list(range(X.shape[0]-window_size)), 20, window_size),
		30000,
		verbose=1
	)

	print(results)

def my_next(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first

def show_loss_and_errors(csv_filename):
	epoch_index_list, train_mape_list, train_mae_list, test_mape_list, test_mae_list, =\
		[], [], [], [], []

	r = reader(open('./csv_train_statistics/'+csv_filename, 'rt' ), delimiter=';')
	my_next(r)
	i = 0
	t = my_next(r)
	# while i != 70 and t != None:
	while t != None:
		epoch_index_list.append(int(t[0]))
		train_mape_list.append(float(t[1]))
		train_mae_list.append(float(t[2]))
		test_mape_list.append(float(t[3]))
		test_mae_list.append(float(t[4]))
		t = my_next(r)
		i+=1

	print(csv_filename + ':')
	print('\ttrain MAPE: ' + str(sum(train_mape_list[-50:])/50))
	print('\ttest MAPE: ' + str(sum(test_mape_list[-50:])/50))
	print('\ttrain MAE: ' + str(sum(train_mae_list[-50:])/50))
	print('\ttest MAE: ' + str(sum(test_mae_list[-50:])/50))
	print()

	plt.subplot(2, 1, 1)
	plt.plot(epoch_index_list, train_mape_list, label='train')
	plt.plot(epoch_index_list, test_mape_list, label='validation')
	plt.ylabel('MAPE')

	plt.subplot(2, 1, 2)
	plt.plot(epoch_index_list, train_mae_list, label='train')
	plt.plot(epoch_index_list, test_mae_list, label='validation')
	plt.xlabel('EPOCHS')
	plt.ylabel('MAE')

	plt.legend()

	# plt.show()
	plt.savefig('./optimization_pictures/' + csv_filename[:-3] + 'png')
	plt.clf()

def predict_only_on_time_model_0(window_size):
	model = Sequential()

	model.add(
		keras.layers.Dense(\
			units=200,
			activation='relu',
			input_shape=(window_size,)
		)
	)

	model.add(
		keras.layers.Dense(\
			units=400,
			activation='relu',
		)
	)

	model.add(
		keras.layers.Dense(\
			units=200,
			activation='relu',
		)
	)

	model.add(
		keras.layers.Dense(\
			units=100,
			activation='relu',
		)
	)

	return model

def generate2(X, index_list, batch_size, window_size):
	while True:
		sub_X = np.empty((\
			batch_size,\
			window_size,\
		))
		sub_y = np.empty((\
			batch_size,\
			window_size,\
		))

		i = 0
		for index in random.sample(index_list,batch_size):
			sub_X[i] = X[index:index+window_size, 0]
			sub_y[i] = X[index:index+window_size, 1]
			i+=1

		yield sub_X, sub_y

def predict_only_on_time_model_1(ws):
	model = Sequential()

	# model.add(
	# 	keras.layers.TimeDistributed(
	# 		keras.layers.Dense(units=4, activation='relu'),
	# 		input_shape=(100, 1)
	# 	)
	# )

	model.add(
		keras.layers.Bidirectional(
			keras.layers.LSTM(units=50,
				return_sequences=True,
			),
			input_shape=(ws, 1),
		)
	)

	# model.add(
	# 	keras.layers.TimeDistributed(
	# 		keras.layers.Dense(units=100,)
	# 	)
	# )

	# model.add(
	# 	keras.layers.LeakyReLU(alpha=0.3)
	# )

	model.add(
		keras.layers.TimeDistributed(
			keras.layers.Dense(units=50,)
		)
	)

	model.add(
		keras.layers.LeakyReLU(alpha=0.3)
	)


	model.add(
		keras.layers.TimeDistributed(
			keras.layers.Dense(units=25,)
		)
	)

	model.add(
		keras.layers.LeakyReLU(alpha=0.3)
	)

	model.add(
		keras.layers.TimeDistributed(
			keras.layers.Dense(units=1, activation='relu')
		)
	)

	return model

def predict_only_on_time_main():

	X = pickle.load(open('./pipe.p','rb'))[:,-2:]

	max_value = -np.ones(2)
	a, b = None, None
	for l in X:
		max_value = np.maximum(max_value, l)
		if a == None or a > l[0]:
			a = l[0]
		if b == None or b > l[1]:
			b = l[1]
	max_value[0] = max_value[0] - a
	max_value[1] = max_value[1] - b
	for i in range(X.shape[0]):
		X[i,0] = 2 * (X[i,0]-a) / max_value[0] - 1
		X[i,1] = (X[i,1]-b) / max_value[1]

	train_index_list, val_index_list = pickle.load(\
		open('train_validation_indexes_lists.p' , 'rb' )
	)

	_, ax = plt.subplots()
	ax.plot(tuple(range(X.shape[0])), X[:,0], 'b+', label='time')
	ax.plot(tuple(range(X.shape[0])), X[:,1], 'r+', label='throughput')
	plt.xlabel('Data Set Index')
	ax.legend()
	plt.show()
	if False:
		plt.clf()
		plt.plot(\
			train_index_list,\
			tuple(\
				map(
					lambda e: X[e,1],
					train_index_list
				)
			),
			'b+'
		)
		plt.plot(\
			val_index_list,\
			tuple(\
				map(
					lambda e: X[e,1],
					val_index_list
				)
			),
			'r+'
		)
		plt.show()

		exit(0)

	if False:
		new_X = np.empty((
			X.shape[0],
			31,
		))
		for i in range(X.shape[0]):
			for p in range(1,31):
				new_X[i, p-1] = X[i,0]**p
			new_X[i,30] = X[i,-1]
		X = new_X


	model = predict_only_on_time_model_1(1000)

	model.compile(
		optimizer=keras.optimizers.Nadam(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	model.summary()

	model.fit_generator(
		generator=generate0(X,train_index_list,32,1000),
		validation_data=generate0(X,val_index_list,32,1000),
		steps_per_epoch=10,
		validation_steps=2,
		epochs=300,
		callbacks=[\
			TensorBoard(log_dir='./board/'),
			# ModelCheckpoint("models/model_{epoch:04d}.hdf5", monitor='loss', period=10),
			# CSVLogger(dump_csv_name, append=True, separator=';'),
		]
	)

def test_generator():

	X = pickle.load(open('./pipe.p','rb'))[:,-2:]

	max_value = -np.ones(2)
	a, b = None, None
	for l in X:
		max_value = np.maximum(max_value, l)
		if a == None or a > l[0]:
			a = l[0]
		if b == None or b > l[1]:
			b = l[1]
	max_value[0] = max_value[0] - a
	max_value[1] = max_value[1] - b

	for i in range(X.shape[0]):
		X[i,0] = 2 * (X[i,0]-a) / max_value[0] - 1
		X[i,1] = (X[i,1]-b) / max_value[1]

	train_index_list, val_index_list = pickle.load(\
		open('train_validation_indexes_lists.p' , 'rb' )
	)

	a, b = next( generate0( X,train_index_list,32,100) )

	# plt.plot(tuple(range(X.shape[0])), X[:,0], 'b+')
	# plt.plot(tuple(range(X.shape[0])), X[:,1], 'r+')

	ind = random.randint(0,31)

	plt.plot(tuple(range(100)), a[ind], 'b+')
	plt.plot(tuple(range(100)), b[ind], 'r+')

	plt.show()

def validate_and_return_encoding(\
	X,
	i,
	failure_name,
	encoding_start,
	encoding_finish,
	):

	a = 0
	have_already_seen_one = False
	j = encoding_start
	while j < encoding_finish:

		if X[i,j] not in (0,1,):
			print(failure_name + ': non encoding value detected !')
			print('\tencoding is between: ' + str(encoding_start) + ' ' + str(encoding_finish))
			print('\tfailed at i j indexes: ' + str(i) + ' ' + str(j))
			print('\tbad values is: ' + str(X[i,j]))
			exit(0)

		if X[i,j] == 1:
			if not have_already_seen_one:
				have_already_seen_one = True
				encoding_value = a
			else:
				print(failure_name + ': second one value detected in the encoding !')
				print('\tencoding is between: ' + str(encoding_start) + ' ' + str(encoding_finish))
				print('\tfailed at i j indexes: ' + str(i) + ' ' + str(j))
				exit(0)
		j += 1
		if not have_already_seen_one: a += 1

	if not have_already_seen_one:
		print(failure_name + ': only zeros in the encoding !')
		print('\tencoding is between: ' + str(encoding_start) + ' ' + str(encoding_finish))
		exit(0)
	return encoding_value

def validate_data_set(cl_num=55, se_num=62):
	'''
	Test if the data set contains bugs.
	'''
	X = pickle.load(open('./pipe.p','rb'))

	index_list = tuple(range(X.shape[0]))

	client_index_array,\
	se_1_index_array,\
	se_2_index_array,\
	se_3_index_array,\
	read_size_array,\
	time_stamp_array,\
	thp_array = np.empty(X.shape[0]),np.empty(X.shape[0]),np.empty(X.shape[0]),\
	np.empty(X.shape[0]),np.empty(X.shape[0]),np.empty(X.shape[0]),np.empty(X.shape[0]),

	se_1_one_hot_start = cl_num
	se_2_one_hot_start = se_1_one_hot_start + se_num
	se_3_one_hot_start = se_2_one_hot_start + se_num

	for i in range(X.shape[0]):

		# TESTING CLIENT ENCODING
		client_index_array[i] =\
			validate_and_return_encoding(X, i, 'Client One-Hot Encoding Failure'\
			, 0, se_1_one_hot_start)

		# TESTING SE_1 ENCODING
		se_1_index_array[i] =\
			validate_and_return_encoding(X, i, 'SE_1 One-Hot Encoding Failure'\
				, se_1_one_hot_start, se_2_one_hot_start)

		# TESTING SE_2 ENCODING
		se_2_index_array[i] =\
			validate_and_return_encoding(X, i, 'SE_2 One-Hot Encoding Failure'\
			, se_2_one_hot_start, se_3_one_hot_start)

		# TESTING SE_3 ENCODING
		se_3_index_array[i] =\
			validate_and_return_encoding(X, i, 'SE_3 One-Hot Encoding Failure'\
			, se_3_one_hot_start, se_3_one_hot_start + se_num)

		# ADDING VALUES INTO PLOT ARRAYS
		read_size_array[i],time_stamp_array[i],thp_array[i] =\
			X[i,-3], X[i,-2], X[i,-1]

	del X

	plt.subplot(7, 1, 1)
	plt.plot(index_list, client_index_array,'b+')

	plt.ylabel('client index')

	plt.subplot(7, 1, 2)
	plt.plot(index_list, se_1_index_array,'b+')

	plt.ylabel('se_1 index')

	plt.subplot(7, 1, 3)
	plt.plot(index_list, se_2_index_array,'b+')

	plt.ylabel('se_2 index')

	plt.subplot(7, 1, 4)
	plt.plot(index_list, se_3_index_array,'b+')

	plt.ylabel('se_3 index')

	plt.subplot(7, 1, 5)
	plt.plot(index_list, read_size_array,'b+')

	plt.ylabel('read size')

	plt.subplot(7, 1, 6)
	plt.plot(index_list, time_stamp_array,'b+')

	plt.ylabel('time stamp')

	plt.subplot(7, 1, 7)
	plt.plot(index_list, thp_array,'b+')

	plt.ylabel('throughput')

	plt.savefig('data_set_fig.png')

	plt.show()

def values_per_thp_plot():
	X = pickle.load(open('./pipe.p','rb'))

	thp_val_dict = dict()

	for l in X:
		if l[-1] not in thp_val_dict:
			thp_val_dict[l[-1]] = 1
		else:
			thp_val_dict[l[-1]] += 1

	keys_list = sorted(list(thp_val_dict.keys()))

	plt.plot(
		keys_list,
		tuple(map(lambda e: thp_val_dict[e], keys_list)),
		'b+'
	)

	plt.show()


if __name__ == '__main__':

	if False:
		dump_train_and_validation_indexes0(0.2, 1000)

	if False:
		for optimizer, opt_string in (\
			(keras.optimizers.Nadam(), 'default_nadam'),
			(keras.optimizers.RMSprop(lr=0.01, rho=0.9), 'rmsp_lr_01'),
			(keras.optimizers.RMSprop(lr=0.001, rho=0.9), 'rmsp_lr_001'),
			(keras.optimizers.RMSprop(lr=0.0001, rho=0.9), 'rmsp_lr_0001'),
			(keras.optimizers.SGD(lr=0.1, momentum=0.0, nesterov=False),'sgd_lr_1_n_False'),
			(keras.optimizers.SGD(lr=0.01, momentum=0.0, nesterov=False),'sgd_lr_01_n_False'),
			(keras.optimizers.SGD(lr=0.001, momentum=0.0, nesterov=False),'sgd_lr_001_n_False'),
			(keras.optimizers.SGD(lr=0.1, momentum=0.0, nesterov=True),'sgd_lr_1_n_True'),
			(keras.optimizers.SGD(lr=0.01, momentum=0.0, nesterov=True),'sgd_lr_01_n_True'),
			(keras.optimizers.SGD(lr=0.001, momentum=0.0, nesterov=True),'sgd_lr_001_n_True'),
			):
			for window_size in (8, 14, 20,):
				train_main_0(\
					optimizer,
					'csv_train_statistics/' + opt_string + '__' + str(window_size) + '.csv',
					window_size,
				)

	if False:
		for fn in listdir('./csv_train_statistics/'):
			show_loss_and_errors(fn)

	if True:
		predict_only_on_time_main()

	if False:
		# train_main_0(\
		# 	keras.optimizers.Nadam(),
		# 	'whatever.csv',
		# 	20,
		# )
		train_main_1()

	if False:
		validate_data_set()

	if False:
		values_per_thp_plot()

	if False:
		test_generator()