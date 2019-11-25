from keras.layers import Concatenate, RepeatVector, TimeDistributed, Reshape, Permute
from keras.layers import Add, Lambda, Flatten, BatchNormalization, Activation
from keras.layers import Input, LSTM, Dense, GRU, Bidirectional, CuDNNLSTM
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
import config

def build_encoder_sz():
	X_depth = config.model_params["encoder_params"]["X_depth"]
	X_size = config.model_params["encoder_params"]["X_size"]
	epsilon_std = config.model_params["encoder_params"]["epsilon_std"]
	phrase_size = config.midi_params["phrase_size"]
	n_cropped_notes = config.midi_params["n_cropped_notes"]
	n_tracks = config.midi_params["n_tracks"]
	s_length = config.model_params["s_length"]
	z_length = config.model_params["z_length"]

	X = Input(shape=(phrase_size, n_cropped_notes, n_tracks), name="X")
	encoder_inputs = X
	
	# X encoder
	h_X = Reshape((phrase_size, n_tracks * n_cropped_notes), name="reshape_X")(X)
	for l in range(X_depth - 1):
		h_X = Bidirectional(
			CuDNNLSTM(X_size, return_sequences=True, name=f"rec_X_{l}"),
			merge_mode="concat", name=f"bidirectional_X_{l}"
		)(h_X)
		#h_X = BatchNormalization(name=f"batchnorm_X_{l}")(h_X)

	h = Bidirectional(
			CuDNNLSTM(X_size, return_sequences=False, name=f"rec_X_{X_depth - 1}"),
			merge_mode="concat", name=f"bidirectional_X_{X_depth - 1}"
		)(h_X)
	#h = BatchNormalization(name=f"batchnorm_X_{X_depth}")(h_X)

	s = Dense(s_length, name="s", activation="sigmoid")(h)

	# reparameterisation trick
	z_mean = Dense(z_length, name='mu', activation='linear')(h)
	z_log_var = Dense(z_length, name='sigma', activation='linear')(h)

	# sampling
	def sampling(args):
		z_mean_, z_log_var_ = args
		batch_size = K.shape(z_mean_)[0]
		epsilon = K.random_normal(shape=(batch_size, z_length), mean=0., stddev=epsilon_std)
		return z_mean_ + K.exp(z_log_var_ / 2) * epsilon
	
	z = Lambda(sampling, output_shape=(z_length,), name='z_sampling')([z_mean, z_log_var])

	encoder_outputs = [s, z]

	return Model(encoder_inputs, encoder_outputs, name="encoder")


def build_encoder_z():
	X_depth = config.model_params["encoder_params"]["X_depth"]
	X_size = config.model_params["encoder_params"]["X_size"]
	epsilon_std = config.model_params["encoder_params"]["epsilon_std"]
	phrase_size = config.midi_params["phrase_size"]
	n_cropped_notes = config.midi_params["n_cropped_notes"]
	n_tracks = config.midi_params["n_tracks"]
	s_length = config.model_params["s_length"]
	z_length = config.model_params["z_length"]

	X = Input(shape=(phrase_size, n_cropped_notes, n_tracks), name="X")
	encoder_inputs = X
	
	# X encoder
	h_X = Reshape((phrase_size, n_tracks * n_cropped_notes), name="reshape_X")(X)
	for l in range(X_depth - 1):
		h_X = Bidirectional(
			CuDNNLSTM(X_size, return_sequences=True, name=f"rec_X_{l}"),
			merge_mode="concat", name=f"bidirectional_X_{l}"
		)(h_X)
		#h_X = BatchNormalization(name=f"batchnorm_X_{l}")(h_X)

	h = Bidirectional(
			CuDNNLSTM(X_size, return_sequences=False, name=f"rec_X_{X_depth - 1}"),
			merge_mode="concat", name=f"bidirectional_X_{X_depth - 1}"
		)(h_X)
	#h = BatchNormalization(name=f"batchnorm_X_{X_depth}")(h_X)

	#s = Dense(s_length, name="s", activation="sigmoid")(h)

	# reparameterisation trick
	z_mean = Dense(z_length, name='mu', activation='linear')(h)
	z_log_var = Dense(z_length, name='sigma', activation='linear')(h)

	# sampling
	def sampling(args):
		z_mean_, z_log_var_ = args
		batch_size = K.shape(z_mean_)[0]
		epsilon = K.random_normal(shape=(batch_size, z_length), mean=0., stddev=epsilon_std)
		return z_mean_ + K.exp(z_log_var_ / 2) * epsilon
	
	z = Lambda(sampling, output_shape=(z_length,), name='z_sampling')([z_mean, z_log_var])

	encoder_outputs = z

	return Model(encoder_inputs, encoder_outputs, name="encoder")