from keras.layers import Concatenate, RepeatVector, TimeDistributed, Reshape, Permute
from keras.layers import Add, Lambda, Flatten, BatchNormalization, Activation
from keras.layers import Input, LSTM, Dense, GRU, Bidirectional, CuDNNLSTM
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
import config

# flat version
def build_decoder_z_flat():
	X_high_depth = config.model_params["decoder_params"]["X_high_depth"]
	X_high_size  = config.model_params["decoder_params"]["X_high_size"]
	X_low_size  = config.model_params["decoder_params"]["X_low_size"]
	phrase_size = config.midi_params["phrase_size"]
	n_cropped_notes = config.midi_params["n_cropped_notes"]
	n_tracks = config.midi_params["n_tracks"]
	s_length = config.model_params["s_length"]
	z_length = config.model_params["z_length"]

	z = Input(shape=(z_length,), name="z")
	decoder_inputs = z

	latent = z

	init_state = Dense(X_high_size, activation="tanh", name="hidden_state_init")(latent)
	
	out_X = []
	for t in range(n_tracks):
		h_X = RepeatVector(phrase_size, name=f"latent_repeat_{t}")(latent)
		for l in range(X_high_depth):
			h_X = CuDNNLSTM(
				X_high_size,
				return_sequences=True,
				#activation="tanh",
				name=f"high_encoder_{t}_{l}"
			)(h_X, initial_state=[init_state, init_state])
		
		out_X_t = TimeDistributed(
			Dense(n_cropped_notes, activation="softmax", name=f"project_out_{t}"),
			name=f"ts_project_{t}"
		)(h_X)

		out_X.append(out_X_t)
	
	decoder_outputs = out_X
	return Model(decoder_inputs, decoder_outputs, name="decoder")


# flat version
def build_decoder_sz_flat():
	X_high_depth = config.model_params["decoder_params"]["X_high_depth"]
	X_high_size  = config.model_params["decoder_params"]["X_high_size"]
	X_low_size  = config.model_params["decoder_params"]["X_low_size"]
	phrase_size = config.midi_params["phrase_size"]
	n_cropped_notes = config.midi_params["n_cropped_notes"]
	n_tracks = config.midi_params["n_tracks"]
	s_length = config.model_params["s_length"]
	z_length = config.model_params["z_length"]

	s = Input(shape=(s_length,), name="s")
	z = Input(shape=(z_length,), name="z")
	decoder_inputs = [s, z]#, Y]

	latent = Concatenate(name="latent_concat")([s, z])

	init_state = Dense(X_high_size, activation="tanh", name="hidden_state_init")(latent)
	
	out_X = []
	for t in range(n_tracks):
		h_X = RepeatVector(phrase_size, name=f"latent_repeat_{t}")(latent)
		for l in range(X_high_depth):
			h_X = CuDNNLSTM(
				X_high_size,
				return_sequences=True,
				#activation="tanh",
				name=f"high_encoder_{t}_{l}"
			)(h_X, initial_state=[init_state, init_state])

		out_X_t = TimeDistributed(
			Dense(n_cropped_notes, activation="softmax", name=f"project_out_{t}"),
			name=f"ts_project_{t}"
		)(h_X)
		out_X.append(out_X_t)
	
	decoder_outputs = out_X
	return Model(decoder_inputs, decoder_outputs, name="decoder")


# hierarchical version
def build_decoder_sz_hierarchical(self):
	X_high_depth = self.decoder_params["X_high_depth"]
	X_high_size  = self.decoder_params["X_high_size"]
	X_low_depth = self.decoder_params["X_low_depth"]
	X_low_size  = self.decoder_params["X_low_size"]
	n_embeddings = self.decoder_params["n_embeddings"]
	
	s = Input(shape=(self.s_length,), name="s")
	z = Input(shape=(self.z_length,), name="z")
	decoder_inputs = [s, z]

	latent = Concatenate(name="latent_concat")([s, z])
	#latent = z
	# get initial state of high decoder
	init_state = Dense(X_high_size, activation="tanh", name="hidden_state_init")(latent)

	out_X = []
	for t in range(self.n_tracks):
		# high decoder produces embeddings
		h_X = RepeatVector(n_embeddings, name=f"latent_repeat_{t}")(latent)

		for l in range(X_high_depth):
			h_X = CuDNNLSTM(
				X_high_size,
				return_sequences=True,
				#activation="tanh",
				name=f"high_encoder_{t}_{l}"
			)(h_X, initial_state=[init_state, init_state])
			
		out_X_t = TimeDistributed(
			LowDecoder(
				output_length=self.subphrase_size,
				timestep_size=self.n_cropped_notes,
				rec_depth=X_low_depth,
				rec_size=X_low_size
			),
			name=f"ts_low_decoder_{t}"
		)(h_X)

		out_X_t = Reshape((self.phrase_size, self.n_cropped_notes), name=f"low_reshape_{t}")(out_X_t)
		out_X.append(out_X_t)


	decoder_outputs = out_X
	return Model(decoder_inputs, decoder_outputs, name="decoder")