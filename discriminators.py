from keras.layers import Input, Dense, Concatenate, Reshape, Bidirectional, CuDNNLSTM
from keras.models import Model

import config

def build_gaussian_discriminator():
	fc_depth = config.model_params["z_discriminator_params"]["fc_depth"]
	fc_size = config.model_params["z_discriminator_params"]["fc_size"]
	z_length = config.model_params["z_length"]

	z = Input(shape=(z_length,), name="z")

	# fully connected layers
	h = z
	for l in range(fc_depth):
		h = Dense(fc_size, activation="tanh", name=f"fc_{l}")(h)
		#h = BatchNormalization(name=f"batchnorm_fc_{l}")(h)
	
	out = Dense(1, activation="linear", name="validity")(h)

	return Model(z, out, name="z_discriminator")

def build_bernoulli_discriminator():
	fc_depth = config.model_params["s_discriminator_params"]["fc_depth"]
	fc_size = config.model_params["s_discriminator_params"]["fc_size"]
	s_length = config.model_params["s_length"]

	s = Input(shape=(s_length,), name="s")

	# fully connected layers
	h = s
	for l in range(fc_depth):
		h = Dense(fc_size, activation="tanh", name=f"fc_{l}")(h)
		#h = BatchNormalization(name=f"batchnorm_fc_{l}")(h)

	out = Dense(1, activation="linear", name="validity")(h)

	return Model(s, out, name="s_discriminator")


def build_gaussian_mixture_discriminator():
	fc_depth = config.model_params["s_discriminator_params"]["fc_depth"]
	fc_size = config.model_params["s_discriminator_params"]["fc_size"]
	z_length = config.model_params["z_length"]
	s_length = config.model_params["s_length"]

	z = Input(shape=(z_length,), name="z")
	y = Input(shape=(s_length,), name="y")

	h = Concatenate(axis=-1, name="concat")([z, y])

	# fully connected layers
	for l in range(fc_depth):
		h = Dense(fc_size, activation="tanh", name=f"fc_{l}")(h)
		#h = BatchNormalization(name=f"batchnorm_fc_{l}")(h)

	out = Dense(1, activation="linear", name="validity")(h)

	return Model([z, y], out, name="s_discriminator")


def build_infomax_network():
	X_depth = config.model_params["infomax_net_params"]["X_depth"]
	X_size = config.model_params["infomax_net_params"]["X_size"]
	phrase_size = config.midi_params["phrase_size"]
	n_cropped_notes = config.midi_params["n_cropped_notes"]
	n_tracks = config.midi_params["n_tracks"]
	s_length = config.model_params["s_length"]
	z_length = config.model_params["z_length"]

	X_drums = Input(shape=(phrase_size, n_cropped_notes, 1), name="X_drums")
	X_bass  = Input(shape=(phrase_size, n_cropped_notes, 1), name="X_bass")
	X_guitar = Input(shape=(phrase_size, n_cropped_notes, 1), name="X_guitar")
	X_strings = Input(shape=(phrase_size, n_cropped_notes, 1), name="X_strings")

	infomax_net_inputs = [X_drums, X_bass, X_guitar, X_strings]

	X = Concatenate(axis=-1, name="concat")([X_drums, X_bass, X_guitar, X_strings])

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

	infomax_net_outputs = s

	return Model(infomax_net_inputs, infomax_net_outputs, name="infomax_network")