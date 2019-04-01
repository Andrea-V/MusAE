from __future__ import print_function, division

from keras.layers import Concatenate, RepeatVector, TimeDistributed, Reshape, Permute
from keras.layers import Add, Lambda, Flatten, BatchNormalization, Activation
from keras.layers import Input, LSTM, Dense, GRU, Bidirectional, CuDNNLSTM
from keras.layers.merge import _Merge

from keras.models import Model
from keras.models import load_model

from keras.optimizers import RMSprop, Adam
from functools import partial

from keras.utils import print_summary, plot_model
from keras.utils import to_categorical   

from keras import backend as K
from keras.engine.topology import Layer

import functools
import tensorflow as tf
import gpustat

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys, os
import numpy as np
import progressbar
import pprint
import time
import itertools
import math, random
import json
from queue import Queue
import threading

import pypianoroll as pproll
import config
from scipy.stats import pearsonr

pp = pprint.PrettyPrinter(indent=4)


class RandomWeightedAverage(_Merge):
	def _merge_function(self, inputs):
		batch_size = K.shape(inputs[0])[0]
		weights = K.random_uniform((batch_size, 1))
		return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class MidiAAE():
	def __init__(self, **kwargs):
		# setting params as class attributes
		self.__dict__.update(kwargs)

		print("n_cropped_notes: ", self.n_cropped_notes)

		# using GPU with most memory avaiable
		self.set_gpu()

		print("Initialising encoder...")
		self.encoder = self.build_encoder_v2()

		print("Initialising decoder...")
		self.decoder = self.build_decoder_v3()

		print("Initialising z discriminator...")
		self.z_discriminator = self.build_z_discriminator()	

		print("Initialising s discriminator...")
		self.s_discriminator = self.build_s_discriminator()

		path = os.path.join(self.plots_path, self.name, "models")
		if not os.path.exists(path):
			os.makedirs(path)

		print("Saving model plots..")
		plot_model(self.encoder, os.path.join(path, "encoder.png"), show_shapes=True)
		plot_model(self.decoder, os.path.join(path, "decoder.png"), show_shapes=True)
		plot_model(self.z_discriminator, os.path.join(path, "z_discriminator.png"), show_shapes=True)
		plot_model(self.s_discriminator, os.path.join(path, "s_discriminator.png"), show_shapes=True)

		#-------------------------------
		# Construct Computational Graph
		# for the Adversarial Autoencoder
		#-------------------------------
		print("Building reconstruction phase's computational graph...")
		self.encoder.trainable = True
		self.decoder.trainable = True
		self.z_discriminator.trainable = False
		self.s_discriminator.trainable = False

		X = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X")

		s_recon, z_recon  = self.encoder(X)
		Y_drums, Y_bass, Y_guitar, Y_strings = self.decoder([s_recon, z_recon])

		self.reconstruction_phase = Model(
			inputs=X,
			outputs=[Y_drums, Y_bass, Y_guitar, Y_strings],
			name="autoencoder"
		)

		plot_model(self.reconstruction_phase, os.path.join(path, "reconstruction_phase.png"), show_shapes=True)

		#-------------------------------
		# Construct Computational Graph
		#    for the z discriminator
		#-------------------------------
		print("Building z regularisation phase's computational graph...")
		self.encoder.trainable = False
		self.decoder.trainable = False
		self.z_discriminator.trainable = True
		self.s_discriminator.trainable = False

		X = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X")
		z_real = Input(shape=(self.z_length,), name="z")

		_, z_fake = self.encoder(X)
		z_int = RandomWeightedAverage(name="weighted_avg_z")([z_real, z_fake])

		z_valid_real = self.z_discriminator(z_real)
		z_valid_fake = self.z_discriminator(z_fake)
		z_valid_int  = self.z_discriminator(z_int)

		self.z_regularisation_phase = Model(
			[z_real, X],
			[z_valid_real, z_valid_fake, z_valid_int, z_int],
			name="z_regularisation_phase"
		)
		plot_model(self.z_regularisation_phase, os.path.join(path, "z_regularisation_phase.png"), show_shapes=True)
		
		#-------------------------------
		# Construct Computational Graph
		#    for the s discriminator
		#-------------------------------
		print("Building s regularisation phase's computational graph...")
		self.encoder.trainable = False
		self.decoder.trainable = False
		self.z_discriminator.trainable = False
		self.s_discriminator.trainable = True

		X = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X")
		s_real = Input(shape=(self.s_length,), name="s")

		s_fake, _ = self.encoder(X)
		s_int = RandomWeightedAverage(name="weighted_avg_s")([s_real, s_fake])

		s_valid_real = self.s_discriminator(s_real)
		s_valid_fake = self.s_discriminator(s_fake)
		s_valid_int  = self.s_discriminator(s_int)

		self.s_regularisation_phase = Model(
			[s_real, X],
			[s_valid_real, s_valid_fake, s_valid_int, s_int],
			name="s_regularisation_phase"
		)

		plot_model(self.s_regularisation_phase, os.path.join(path, "s_regularisation_phase.png"), show_shapes=True)

		#-------------------------------
		# Construct Computational Graph
		# for the generator (encoder)
		#-------------------------------
		print("Building generator regularisation phase's computational graph...")
		self.encoder.trainable = True
		self.decoder.trainable = False
		self.z_discriminator.trainable = False
		self.s_discriminator.trainable = False

		X = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X")

		s_gen, z_gen  = self.encoder(X)
		
		z_valid_gen = self.z_discriminator(z_gen)
		s_valid_gen = self.s_discriminator(s_gen)

		self.gen_regularisation_phase = Model(
			inputs=X,
			outputs=[s_valid_gen, z_valid_gen],
			name="gen_regularisation_phase"
		)
		plot_model(self.gen_regularisation_phase, os.path.join(path, "gen_regularisation_phase.png"), show_shapes=True)

		#-------------------------------
		# Construct Computational Graph
		# for the supervised phase
		#-------------------------------
		print("Building supervised phase's computational graph...")
		self.encoder.trainable = True
		self.decoder.trainable = False
		self.z_discriminator.trainable = False
		self.s_discriminator.trainable = False

		X = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X")

		s_pred, _  = self.encoder(X)

		self.supervised_phase = Model(
			inputs=X,
			outputs=s_pred,
			name="supervised_phase"
		)

		plot_model(self.supervised_phase, os.path.join(path, "supervised_phase.png"), show_shapes=True)

		#-------------------------------
		# Construct Computational Graph
		# for the generator (encoder)
		#-------------------------------
		print("Building adversarial autoencoder's computational graph...")
		self.encoder.trainable = True
		self.decoder.trainable = True
		self.z_discriminator.trainable = True
		self.s_discriminator.trainable = True

		X = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X")
		z_real = Input(shape=(self.z_length,), name="z")
		s_real = Input(shape=(self.s_length,), name="s")

		Y_drums, Y_bass, Y_guitar, Y_strings = self.reconstruction_phase(X)
		z_valid_real, z_valid_fake, z_valid_int, z_int = self.z_regularisation_phase([z_real, X])
		s_valid_real, s_valid_fake, s_valid_int, s_int = self.s_regularisation_phase([s_real, X])
		s_valid_gen, z_valid_gen = self.gen_regularisation_phase(X)
		s_pred = self.supervised_phase(X)

		self.adversarial_autoencoder = Model(
			inputs=[s_real, z_real, X],
			outputs=[
				Y_drums, Y_bass, Y_guitar, Y_strings,
				s_valid_real, s_valid_fake, s_valid_int,
				z_valid_real, z_valid_fake, z_valid_int,
				s_valid_gen, z_valid_gen,
				s_pred
			],
			name="adversarial_autoencoder"
		)

		# prepare gp losses
		self.s_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=s_int)
		self.s_gp_loss.__name__ = "gradient_penalty_s"
		
		self.z_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=z_int)
		self.z_gp_loss.__name__ = "gradient_penalty_z"

		self.adversarial_autoencoder.compile(
			loss=[
				"categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy",
				self.wasserstein_loss, self.wasserstein_loss, self.s_gp_loss,
				self.wasserstein_loss, self.wasserstein_loss, self.z_gp_loss,
				self.wasserstein_loss, self.wasserstein_loss,
				"binary_crossentropy"
			],
			loss_weights=[
				self.reconstruction_weight, self.reconstruction_weight, self.reconstruction_weight, self.reconstruction_weight,
				self.supervised_weight, self.supervised_weight, self.supervised_weight * self.s_lambda,
				self.regularisation_weight, self.regularisation_weight, self.regularisation_weight * self.z_lambda,
				self.supervised_weight, self.regularisation_weight,
				self.supervised_weight
			],
			optimizer=self.aae_optim,
			metrics=[
				"categorical_accuracy",
				"binary_accuracy"
			]
		)
		plot_model(self.adversarial_autoencoder, os.path.join(path, "adversarial_autoencoder.png"), show_shapes=True)

	def set_gpu(self):
		stats = gpustat.GPUStatCollection.new_query()
		ids = map(lambda gpu: int(gpu.entry['index']), stats)
		ratios = map(lambda gpu: float(gpu.entry['memory.used']) / float(gpu.entry['memory.total']), stats)
		bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
	 
		print("Setting GPU to: {}".format(bestGPU))
		os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
		os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)

	# Just report the mean output of the model (useful for WGAN)
	def output(self, y_true, y_pred):
		return K.mean(y_pred)

	# wrapper for using tensorflow metrics in keras
	def as_keras_metric(self, method):
		@functools.wraps(method)
		def wrapper(self, args, **kwargs):
			""" Wrapper for turning tensorflow metrics into keras metrics """
			value, update_op = method(self, args, **kwargs)
			K.get_session().run(tf.local_variables_initializer())
			with tf.control_dependencies([update_op]):
				value = tf.identity(value)
			return value
		return wrapper

	def precision(self, y_true, y_pred):
		# true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  
		# predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))  
		# precision = true_positives / (predicted_positives + K.epsilon())    
		# return precision
		precision = self.as_keras_metric(tf.metrics.precision)
		return precision(y_true, y_pred)

	def recall(self, y_true, y_pred):
		# true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  
		# possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))   
		# recall = true_positives / (possible_positives + K.epsilon())    
		recall = self.as_keras_metric(tf.metrics.recall)
		return recall(y_true, y_pred)

	def f1_score(self, y_true, y_pred):
		precision = self.as_keras_metric(tf.metrics.precision)
		recall = self.as_keras_metric(tf.metrics.recall)

		p = precision(y_true, y_pred)
		r = recall(y_true, y_pred)
		return (2 * p * r) / (p + r + K.epsilon())

	# dummy loss
	def no_loss(self, y_true, y_pred):
		return K.zeros(shape=(1,))

	def wasserstein_loss(self, y_true, y_pred):
		return K.mean(y_true * y_pred)

	# def wasserstein_loss_real(self, y_true, y_pred):
	#     return -K.mean(y_pred)

	# def wasserstein_loss_fake(self, y_true, y_pred):
	#     return K.mean(y_pred)

	def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
		def _compute_gradients(tensor, var_list):
			grads = tf.gradients(tensor, var_list)
			return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

		#gradients = K.gradients(y_pred, averaged_samples)[0]
		gradients = _compute_gradients(y_pred, [averaged_samples])[0]
		gradients_sqr = K.square(gradients)
		gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
		gradient_l2_norm = K.sqrt(gradients_sqr_sum)
		gradient_penalty = K.square(1 - gradient_l2_norm)
		return K.mean(gradient_penalty)

	def build_encoder_v2(self):
		X_depth = self.encoder_params["X_depth"]
		X_size = self.encoder_params["X_size"]
		epsilon_std = self.encoder_params["epsilon_std"]

		X = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X")
		encoder_inputs = X
		
		# X encoder
		h_X = Reshape((self.phrase_size, self.n_tracks * self.n_cropped_notes), name="reshape_X")(X)
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

		s = Dense(self.s_length, name="s", activation="sigmoid")(h)

		# reparameterisation trick
		z_mean = Dense(self.z_length, name='mu', activation='linear')(h)
		z_log_var = Dense(self.z_length, name='sigma', activation='linear')(h)

		# sampling
		z_length = self.z_length # needed to pickle model, not sure why but otherwise it doesn't work
		def sampling(args):
			z_mean_, z_log_var_ = args
			batch_size = K.shape(z_mean_)[0]
			epsilon = K.random_normal(shape=(batch_size, z_length), mean=0., stddev=epsilon_std)
			return z_mean_ + K.exp(z_log_var_ / 2) * epsilon
		
		z = Lambda(sampling, output_shape=(self.z_length,), name='z_sampling')([z_mean, z_log_var])

		#assert(self.z_length >= self.s_length)

		# selecting first s_length latent factors as genre representation
		#s = Lambda(lambda x: x[:,:self.s_length], name="s")(z)
		#s = Activation("softmax", name="s_softmax")(s)

		encoder_outputs = [s, z]

		return Model(encoder_inputs, encoder_outputs, name="encoder")

	def build_decoder_v3(self):
		X_high_depth = self.decoder_params["X_high_depth"]
		X_high_size  = self.decoder_params["X_high_size"]
		X_low_size  = self.decoder_params["X_low_size"]
		n_embeddings = self.decoder_params["n_embeddings"]
		teacher_forcing = self.decoder_params["teacher_forcing"] # TODO

		#print("- Initialising high decoder...")
		#Y = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="teacher_forcing")
		s = Input(shape=(self.s_length,), name="s")
		z = Input(shape=(self.z_length,), name="z")
		decoder_inputs = [s, z]#, Y]

		latent = Concatenate(name="latent_concat")([s, z])
		#latent = z
		# get initial state of high decoder
		init_state = Dense(X_high_size, activation="tanh", name="hidden_state_init")(latent)
		
		out_X = []
		for t in range(self.n_tracks):
			# high decoder produces embeddings
			h_X = RepeatVector(self.phrase_size, name=f"latent_repeat_{t}")(latent)
			for l in range(X_high_depth):
				h_X = CuDNNLSTM(
					X_high_size,
					return_sequences=True,
					#activation="tanh",
					name=f"high_encoder_{t}_{l}"
				)(h_X, initial_state=[init_state, init_state])
			
			#print("h_X:", h_X.shape)

			out_X_t = TimeDistributed(
				Dense(self.n_cropped_notes, activation="softmax", name=f"project_out_{t}"),
				name=f"ts_project_{t}"
			)(h_X)

			#print("out_X_t:", out_X_t.shape)

			out_X.append(out_X_t)

		#print("out_X:\n", out_X)
		#exit(0)
		
		decoder_outputs = out_X
		return Model(decoder_inputs, decoder_outputs, name="decoder")

	def build_z_discriminator(self):
		fc_depth = self.z_discriminator_params["fc_depth"]
		fc_size = self.z_discriminator_params["fc_size"]

		z = Input(shape=(self.z_length,), name="z")

		# fully connected layers
		h = z
		for l in range(fc_depth):
			h = Dense(fc_size, activation="tanh", name=f"fc_{l}")(h)
			#h = BatchNormalization(name=f"batchnorm_fc_{l}")(h)
		
		out = Dense(1, activation="linear", name="validity")(h)

		return Model(z, out, name="z_discriminator")

	def build_s_discriminator(self):
		fc_depth = self.s_discriminator_params["fc_depth"]
		fc_size = self.s_discriminator_params["fc_size"]

		s = Input(shape=(self.s_length,), name="s")

		# fully connected layers
		h = s
		for l in range(fc_depth):
			h = Dense(fc_size, activation="tanh", name=f"fc_{l}")(h)
			#h = BatchNormalization(name=f"batchnorm_fc_{l}")(h)

		out = Dense(1, activation="linear", name="validity")(h)

		return Model(s, out, name="s_discriminator")


	def train_v2(self, dataset):
		epsilon_std = self.encoder_params["epsilon_std"]
		# create checkpoint and plots folder
		now = str(int(round(time.time())))

		paths = {
			"interpolations": os.path.join(self.interpolations_path, self.name),
			"autoencoded": os.path.join(self.autoencoded_path, self.name),
			"checkpoints": os.path.join(self.checkpoints_path, self.name),
			"plots": os.path.join(self.plots_path, self.name),
			"sampled": os.path.join(self.sampled_path, self.name),
			"style_transfers": os.path.join(self.style_transfers_path, self.name),
			"latent_sweeps": os.path.join(self.latent_sweeps_path, self.name)
		}
		for key in paths:
			if not os.path.exists(paths[key]):
				os.makedirs(paths[key])
		
		print("Splitting training set and validation set...")
		batches_path = os.path.join(self.dataset_path, "batches", "X")

		_, _, files = next(os.walk(batches_path))
		self.len_dataset = len(files)
		
		tr_set, vl_set = train_test_split(files, test_size=self.test_size)
		del files

		self.len_tr_set = len(tr_set)
		self.len_vl_set = len(vl_set)

		print("Number of TR batches:", self.len_tr_set)
		print("Number of VL batches:", self.len_vl_set)
		# storing losses over time
		#len_z_losses = #len(self.z_regularisation_phase.metrics_names) 
		#len_s_losses = #len(self.s_regularisation_phase.metrics_names)
		#len_gen_losses = #len(self.gen_regularisation_phase.metrics_names)
		#len_aae_losses = #len(self.reconstruction_phase.metrics_names)

		z_losses_tr = []#np.zeros((self.n_epochs * self.len_tr_set, len_z_losses))
		s_losses_tr = []#np.zeros((self.n_epochs * self.len_tr_set, len_s_losses))
		gen_losses_tr = []#np.zeros((self.n_epochs * self.len_tr_set, len_gen_losses))
		aae_losses_tr = []#np.zeros((self.n_epochs * self.len_tr_set, len_aae_losses))
		aae_accuracies_tr = []
		supervised_losses_tr = []
		supervised_accuracies_tr = []

		z_losses_vl = []#np.zeros((self.n_epochs * self.len_vl_set, len_z_losses))
		s_losses_vl = []#np.zeros((self.n_epochs * self.len_vl_set, len_s_losses))
		gen_losses_vl = []#np.zeros((self.n_epochs * self.len_tr_set, len_gen_losses))
		aae_losses_vl = []#np.zeros((self.n_epochs * self.len_vl_set, len_aae_losses))
		aae_accuracies_vl = []#np.zeros((self.n_epochs * self.len_vl_set, len_aae_losses))
		supervised_losses_vl = []
		supervised_accuracies_vl = []

		#... let the training begin!
		bar = progressbar.ProgressBar(max_value=(self.n_epochs * self.len_dataset))
		pbc = 0
		pbc_tr = 0
		pbc_vl = 0
		annealing_first_stage = False
		annealing_second_stage = False
		annealing_third_stage = False
		annealing_fourth_stage = False
		annealing_fifth_stage = False
		annealing_sixth_stage = False
		annealing_seventh_stage = False
		annealing_eighth_stage = False
		annealing_ninth_stage = False
		annealing_tenth_stage = False
		#bar.update(0)
		for epoch in range(self.n_epochs):
			print("- Epoch", epoch+1, "of", self.n_epochs)
			print("-- Number of TR batches:", self.len_tr_set)
			print("-- Number of VL batches:", self.len_vl_set)
			epoch_pbc = pbc

			print("Generating training batches...")
			tr_queue = Queue(maxsize=128)

			def async_batch_generator_tr():
				#training_set = dataset.generate_batches(pianorolls_path, tr_set, batch_size=self.batch_size)
				tr_batches = list(range(self.len_tr_set))
				random.shuffle(tr_batches)
				for i in tr_batches:
					#tr_queue.put(dataset.preprocess(next(training_set)), block=True)
					tr_queue.put(dataset.select_batch(i), block=True)

			training_batch_thread = threading.Thread(target=async_batch_generator_tr)
			training_batch_thread.start()

			print("Training on training set...")
			# train on the training set
			for _ in range(self.len_tr_set):
				bar.update(pbc)
				
				#start = time.time()
				X, Y, label = tr_queue.get(block=True)
				label = label[:, :self.s_length]
				#end = time.time()
				#print("time elapsed in queue: ", end - start)

				n_chunks = X.shape[0]

				# Adversarial ground truth (wasserstein)
				real_gt  =  -np.ones((n_chunks, 1))
				fake_gt  =  np.ones((n_chunks, 1))
				dummy_gt =  np.zeros((n_chunks, 1)) # Dummy gt for gradient penalty (not actually used)

				# draw z from N(0,epsilon_std)
				z = np.random.normal(0, epsilon_std, (n_chunks, self.z_length))
				# draw s from Cat(s_length)
				# categorical = multinomial with k>2 and n=1
				k = self.s_length
				# s is a k-hot vector of tags
				s = np.random.multinomial(k, [1/k]*k, n_chunks)
				s[s > 1] = 1

				#Y_split = [ Y[:, :, : , t] for t in range(self.n_tracks) ]
				Y_drums = Y[:, :, : , 0]
				Y_bass = Y[:, :, : , 1]
				Y_guitar = Y[:, :, : , 2]
				Y_strings = Y[:, :, : , 3]
				aae_loss = self.adversarial_autoencoder.train_on_batch(
					[s, z, X],
					[
						Y_drums, Y_bass, Y_guitar, Y_strings,
						real_gt, fake_gt, dummy_gt,
						real_gt, fake_gt, dummy_gt,
						real_gt, real_gt,
						label
					]
				)
				
				# pp.pprint(list(enumerate(self.adversarial_autoencoder.metrics_names)))
				# exit(0)
				
				aae_losses_tr.append(aae_loss[1:5])
				aae_accuracies_tr.append([aae_loss[14], aae_loss[16], aae_loss[18], aae_loss[20]])
				s_losses_tr.append(aae_loss[5:8])
				z_losses_tr.append(aae_loss[8:11])
				gen_losses_tr.append(aae_loss[11:13])
				supervised_losses_tr.append([aae_loss[13]])
				supervised_accuracies_tr.append([aae_loss[39]])

				#print(self.adversarial_autoencoder.metrics_names)

				if pbc_tr % 5000 == 0:
					print("\nInterpolating bars...")
					#self.save_interpolation(paths["interpolations"], dataset, 50, 12345, str(pbc_tr))
					#self.save_interpolation(paths["interpolations"], dataset, 50, 129000, str(pbc_tr))
					#self.save_interpolation(paths["interpolations"], dataset, 35002, 5201, str(pbc_tr))
					#self.save_interpolation(paths["interpolations"], dataset, 35002, 99221, str(pbc_tr))
					#self.save_interpolation(paths["interpolations"], dataset, 70000, 25201, str(pbc_tr))
					#self.save_interpolation(paths["interpolations"], dataset, 70000, 111000, str(pbc_tr))
					
					print("\nLatent space Z sweeps...")
					#self.save_z_latents_sweep(paths["latent_sweeps"], dataset, 50, str(pbc_tr))
					#self.save_z_latents_sweep(paths["latent_sweeps"], dataset, 70000, str(pbc_tr))
					#self.save_z_latents_sweep(paths["latent_sweeps"], dataset, 111000, str(pbc_tr))
					#self.save_z_latents_sweep(paths["latent_sweeps"], dataset, 1592, str(pbc_tr))
					#self.save_z_latents_sweep(paths["latent_sweeps"], dataset, 23001, str(pbc_tr))

					print("\nAutoencoding songs...")
					# # honesty
					#self.save_autoencoded_song(paths["autoencoded"], dataset, 53, str(pbc_tr))
					# breaking the law
					#self.save_autoencoded_song(paths["autoencoded"], dataset, 210, str(pbc_tr))
					# # ticket to ride
					#self.save_autoencoded_song(paths["autoencoded"], dataset, 257, str(pbc_tr))
					# # house of the rising sun
					#self.save_autoencoded_song(paths["autoencoded"], dataset, 318, str(pbc_tr))
					# # brain damage
					#self.save_autoencoded_song(paths["autoencoded"], dataset, 406, str(pbc_tr))
					# # sweet child o mine
					#self.save_autoencoded_song(paths["autoencoded"], dataset, 577, str(pbc_tr))
					# # mamma maria
					#self.save_autoencoded_song(paths["autoencoded"], dataset, 815, str(pbc_tr))
					# bille jean
					#self.save_autoencoded_song(paths["autoencoded"], dataset, 1993, str(pbc_tr))
					# # lady marmalade
					#self.save_autoencoded_song(paths["autoencoded"], dataset, 946, str(pbc_tr))
					# # samba pa ti
					#self.save_autoencoded_song(paths["autoencoded"], dataset, 992, str(pbc_tr))
					# # behind blue eyes
					#self.save_autoencoded_song(paths["autoencoded"], dataset, 1119, str(pbc_tr))

					print("\nSampling new phrases...")
					# self.save_sampled_phrase(paths["sampled"], dataset, epsilon_std, str(pbc_tr) + "_0")
					# self.save_sampled_phrase(paths["sampled"], dataset, 2*epsilon_std, str(pbc_tr) + "_1")
					# self.save_sampled_phrase(paths["sampled"], dataset, 5*epsilon_std, str(pbc_tr) + "_2")

					print("\nPerforming style transfer...")
					# abba - hasta manana
					#self.style_transfer(paths["style_transfers"], dataset, 1190, str(pbc_tr))
					# # queen - fat bottomed girls
					#self.style_transfer(paths["style_transfers"], dataset, 2932, str(pbc_tr))
					# # Ricky Valance - Fly Me To The Moon
					#self.style_transfer(paths["style_transfers"], dataset, 2853, str(pbc_tr))
					# 2454 -> Madness - One Step Beyond
					#self.style_transfer(paths["style_transfers"], dataset, 2454, str(pbc_tr))
					# 2148 -> The Police - Walking On The Moon
					#self.style_transfer(paths["style_transfers"], dataset, 2149, str(pbc_tr))
					# # 1979 -> The Police - Spirits In The Material World
					#self.style_transfer(paths["style_transfers"], dataset, 1979, str(pbc_tr))
					# # 1882 -> Robert Palmer - Bad Case Of Loving You (Doctor_ Doctor)
					#self.style_transfer(paths["style_transfers"], dataset, 1882, str(pbc_tr))
					# # 657 -> Elton John - Spirit In The Sky
					#self.style_transfer(paths["style_transfers"], dataset, 657, str(pbc_tr))
					# # 2112 -> ZZ Top - Just Got Paid
					#self.style_transfer(paths["style_transfers"], dataset, 2112, str(pbc_tr))

					print("TR batches in the queue: ", tr_queue.qsize())

				if pbc_tr % 5000 == 0:
					print("\nPlotting stats...")
					print("Regularisation weight:", K.get_value(self.regularisation_weight))
					self.plot_timewise_stats(
						paths["plots"], "TR", 0, pbc_tr+1,
						np.array(z_losses_tr),
						np.array(s_losses_tr),
						np.array(gen_losses_tr),
						np.array(aae_losses_tr),
						np.array(aae_accuracies_tr),
						np.array(supervised_losses_tr),
						np.array(supervised_accuracies_tr),
					)
					print("TR batches in the queue: ", tr_queue.qsize())

				# annealing the regularisation part
				if pbc_tr > 5000 and not annealing_first_stage:
					K.set_value(self.regularisation_weight, 0.1)
					print("Regularisation weight annealed to ", K.get_value(self.regularisation_weight))
					annealing_first_stage = True
				elif pbc_tr > 10000 and not annealing_second_stage:
					K.set_value(self.regularisation_weight, 0.2)
					print("Regularisation weight annealed to ", K.get_value(self.regularisation_weight))
					annealing_second_stage = True
				elif pbc_tr > 15000 and not annealing_third_stage:
					K.set_value(self.regularisation_weight, 0.3)
					print("Regularisation weight annealed to ", K.get_value(self.regularisation_weight))
					annealing_third_stage = True
				elif pbc_tr > 20000 and not annealing_fourth_stage:
					K.set_value(self.regularisation_weight, 0.4)
					print("Regularisation weight annealed to ", K.get_value(self.regularisation_weight))
					annealing_fourth_stage = True
				elif pbc_tr > 25000 and not annealing_fifth_stage:
					K.set_value(self.regularisation_weight, 0.5)
					print("Regularisation weight annealed to ", K.get_value(self.regularisation_weight))
					annealing_fifth_stage = True
				elif pbc_tr > 30000 and not annealing_sixth_stage:
					K.set_value(self.regularisation_weight, 0.6)
					print("Regularisation weight annealed to ", K.get_value(self.regularisation_weight))
					annealing_sixth_stage = True
				elif pbc_tr > 35000 and not annealing_seventh_stage:
					K.set_value(self.regularisation_weight, 0.7)
					print("Regularisation weight annealed to ", K.get_value(self.regularisation_weight))
					annealing_seventh_stage = True
				elif pbc_tr > 40000 and not annealing_eighth_stage:
					K.set_value(self.regularisation_weight, 0.8)
					print("Regularisation weight annealed to ", K.get_value(self.regularisation_weight))
					annealing_eighth_stage = True
				elif pbc_tr > 45000 and not annealing_ninth_stage:
					K.set_value(self.regularisation_weight, 0.9)
					print("Regularisation weight annealed to ", K.get_value(self.regularisation_weight))
					annealing_ninth_stage = True
				elif pbc_tr > 50000 and not annealing_tenth_stage:
					K.set_value(self.regularisation_weight, 1)
					print("Regularisation weight annealed to ", K.get_value(self.regularisation_weight))
					annealing_tenth_stage = True

				pbc += 1
				pbc_tr += 1
			# at the end of each epoch, we evaluate on the validation set
			print("Generating validation batches...")
			vl_queue = Queue(maxsize=128)

			def async_batch_generator_vl():
				#training_set = dataset.generate_batches(pianorolls_path, tr_set, batch_size=self.batch_size)
				vl_batches = range(self.len_vl_set)
				#random.shuffle(vl_batches)
				for i in vl_batches:
					#tr_queue.put(dataset.preprocess(next(training_set)), block=True)
					vl_queue.put(dataset.select_batch(i), block=True)

			validation_batch_thread = threading.Thread(target=async_batch_generator_vl)
			validation_batch_thread.start()

			print("\nEvaluating on validation set...")
			# evaluating on validation set
			pbc_vl0 = pbc_vl

			# generate batches for validation set as well
			for _ in range(self.len_vl_set):
				bar.update(pbc)
				# try:
				X, Y, label = vl_queue.get(block=True)
				label = label[:, :self.s_length]
				#print("batch get")
				
				n_chunks = X.shape[0]

				# Adversarial ground truths (wasserstein)
				real_gt = -np.ones((n_chunks, 1))
				fake_gt = np.ones((n_chunks, 1))
				dummy_gt = np.zeros((n_chunks, 1)) # Dummy gt for gradient penalty (not actually used)

				# draw z from N(0,epsilon_std)
				z = np.random.normal(0, epsilon_std, (n_chunks, self.z_length))
				# draw s from Cat(s_length)
				# categorical = multinomial with k>2 and n=1
				k = self.s_length
				s = np.random.multinomial(k, [1/k]*k, n_chunks)
				s[s > 1] = 1
				
				#Y_split = [ Y[:, :, : , t] for t in range(self.n_tracks) ]
				Y_drums = Y[:, :, : , 0]
				Y_bass = Y[:, :, : , 1]
				Y_guitar = Y[:, :, : , 2]
				Y_strings = Y[:, :, : , 3]
				aae_loss = self.adversarial_autoencoder.test_on_batch(
					[s, z, X],
					[
						Y_drums, Y_bass, Y_guitar, Y_strings,
						real_gt, fake_gt, dummy_gt,
						real_gt, fake_gt, dummy_gt,
						real_gt, real_gt,
						label
					]
				)

				aae_losses_vl.append(aae_loss[1:5])
				aae_accuracies_vl.append([aae_loss[14], aae_loss[16], aae_loss[18], aae_loss[20]])
				s_losses_vl.append(aae_loss[5:8])
				z_losses_vl.append(aae_loss[8:11])
				gen_losses_vl.append(aae_loss[11:13])
				supervised_losses_vl.append([aae_loss[13]])
				supervised_accuracies_vl.append([aae_loss[39]])

				pbc += 1
				pbc_vl += 1

			# end epoch
			# computing loss for this epoch VL
			print("\n- Epoch end.")
			print("-- Mean s discriminator loss VL: ", np.array(s_losses_vl)[pbc_vl0+1:pbc_vl, :].mean(), "±", np.array(s_losses_vl)[pbc_vl0+1:pbc_vl, :].std())
			print("-- Mean z discriminator loss VL: ", np.array(z_losses_vl)[pbc_vl0+1:pbc_vl, :].mean(), "±", np.array(z_losses_vl)[pbc_vl0+1:pbc_vl, :].std())
			print("-- Mean generator loss VL: ", np.array(gen_losses_vl)[pbc_vl0+1:pbc_vl, :].mean(), "±", np.array(gen_losses_vl)[pbc_vl0+1:pbc_vl, :].std())
			print("-- Mean autoencoder loss VL: ", np.array(aae_losses_vl)[pbc_vl0+1:pbc_vl, :].mean(), "±", np.array(aae_losses_vl)[pbc_vl0+1:pbc_vl, :].std())
			print("-- Mean supervised loss VL: ", np.array(supervised_losses_vl)[pbc_vl0+1:pbc_vl, :].mean(), "±", np.array(supervised_losses_vl)[pbc_vl0+1:pbc_vl, :].std())

			print("-- Mean drums accuracy VL: ", np.array(aae_accuracies_vl)[pbc_vl0+1:pbc_vl, 0].mean(), "±", np.array(aae_accuracies_vl)[pbc_vl0+1:pbc_vl, 0].std())
			print("-- Mean bass accuracy VL: ", np.array(aae_accuracies_vl)[pbc_vl0+1:pbc_vl, 1].mean(), "±", np.array(aae_accuracies_vl)[pbc_vl0+1:pbc_vl, 1].std())
			print("-- Mean guitar accuracy VL: ", np.array(aae_accuracies_vl)[pbc_vl0+1:pbc_vl, 2].mean(), "±", np.array(aae_accuracies_vl)[pbc_vl0+1:pbc_vl, 2].std())
			print("-- Mean strings accuracy VL: ", np.array(aae_accuracies_vl)[pbc_vl0+1:pbc_vl, 3].mean(), "±", np.array(aae_accuracies_vl)[pbc_vl0+1:pbc_vl, 3].std())

			print("-- Mean autoencoder accuracy VL: ", np.array(aae_accuracies_vl)[pbc_vl0+1:pbc_vl, :].mean(), "±", np.array(aae_accuracies_vl)[pbc_vl0+1:pbc_vl, :].std())
			print("-- Mean supervised accuracy VL: ", np.array(supervised_accuracies_vl)[pbc_vl0+1:pbc_vl, :].mean(), "±", np.array(supervised_accuracies_vl)[pbc_vl0+1:pbc_vl, :].std())

			recon_loss_mean = np.array(aae_losses_vl)[pbc_vl0+1:pbc_vl, 0].mean()
			recon_loss_std = np.array(aae_losses_vl)[pbc_vl0+1:pbc_vl, 0].std()

			print("Plotting VL stats...")
			self.plot_timewise_stats(
				paths["plots"], "VL", 0, pbc_vl,
				np.array(z_losses_vl),
				np.array(s_losses_vl),
				np.array(gen_losses_vl),
				np.array(aae_losses_vl),
				np.array(aae_accuracies_vl),
				np.array(supervised_losses_vl),
				np.array(supervised_accuracies_vl),
			)

			print("Saving validation loss...")
			

	def autoencode(self, X, dataset):
		X, Y = dataset.preprocess(X)
		s, z = self.encoder.predict(X)
		X_drums, X_bass, X_guitar, X_strings = self.decoder.predict([s, z])
		X = dataset.postprocess(X_drums, X_bass, X_guitar, X_strings)
		return s, z, X

	def plot_timewise_stats(self, path, prefix, t_0, t, z_losses, s_losses, gen_losses, aae_losses, aae_accuracies, sup_losses, sup_accuracies):
		assert(t > 0)
		ts = np.arange(t_0, t)

		xs = ts
		def single_plot(metrics, colours, labels, savename, legend=True):
			assert(len(metrics) == len(colours)) 
			assert(len(metrics) == len(labels)) 

			for metric, colour, label in zip(metrics, colours, labels):
				plt.plot(xs, metric, colour, label=label)

			if legend:
				plt.legend()

			#plt.tight_layout()
			plt.savefig(os.path.join(path, prefix + savename))
			plt.clf()

		plt.figure(figsize=(30, 10))

		# Z plots
		metrics = [z_losses[ts, 0], z_losses[ts, 1]]
		colours = ["g", "r"]
		labels  = ["z loss real", "z loss fake"]
		single_plot(metrics, colours, labels, "_z_loss")

		metrics = [z_losses[ts, 2]]
		colours = ["b"]
		labels  = ["z gradient penalty"]
		single_plot(metrics, colours, labels, "_z_loss_gradient_penalty")

		# S plots
		metrics = [s_losses[ts, 0], s_losses[ts, 1]]
		colours = ["g", "r"]
		labels  = ["s loss real", "s loss fake"]
		single_plot(metrics, colours, labels, "_s_loss")

		metrics = [s_losses[ts, 2]]
		colours = ["b"]
		labels  = ["s gradient penalty"]
		single_plot(metrics, colours, labels, "_s_loss_gradient_penalty")

		# Generator plots
		metrics = [gen_losses[ts, 0], gen_losses[ts, 1]]
		colours = ["b", "orange"]
		labels  = ["s loss", "z loss"]
		single_plot(metrics, colours, labels, "_generator_loss")

		# supervised plots
		metrics = [sup_losses[ts, 0]]
		colours = ["r"]
		labels = ["s supervised loss"]
		single_plot(metrics, colours, labels, "_s_supervised_loss")

		metrics = [sup_accuracies[ts, 0]]
		colours = ["orange"]
		labels  = ["s supervised loss"]
		single_plot(metrics, colours, labels, "_s_supervised_accuracy")

		# AAE plots
		metrics = [
			aae_losses[ts, 0],
			aae_losses[ts, 1],
			aae_losses[ts, 2],
			aae_losses[ts, 3],
			#aae_losses[ts, 4]
		]
		colours = ["b", "r", "g", "orange"]
		labels  = [
			"AE reconstruction loss - drums",
			"AE reconstruction loss - guitar",
			"AE reconstruction loss - bass",
			"AE reconstruction loss - strings"
		]
		single_plot(metrics, colours, labels, "_aae_reconstruction")

		metrics = [aae_accuracies[ts, 0]]
		colours = ["orange"]
		labels  = ["accuracy"]
		single_plot(metrics, colours, labels, "_aae_reconstruction_metrics_drums")

		metrics = [aae_accuracies[ts, 1]]
		colours = ["orange"]
		labels  = ["accuracy"]
		single_plot(metrics, colours, labels, "_aae_reconstruction_metrics_guitar")

		metrics = [aae_accuracies[ts, 2]]
		colours = ["orange"]
		labels  = ["accuracy"]
		single_plot(metrics, colours, labels, "_aae_reconstruction_metrics_bass")

		metrics = [aae_accuracies[ts, 3]]
		colours = ["orange"]
		labels  = ["accuracy"]
		single_plot(metrics, colours, labels, "_aae_reconstruction_metrics_strings")

		plt.close(plt.gcf()) # deallocate current figure from memory


	def save_sampled_phrase(self, path, dataset, epsilon_std, prefix):
		# draw z from N(0,epsilon_std)
		z = np.random.normal(0, epsilon_std, (32, self.z_length))
		# draw s from Cat(s_length)
		# categorical = multinomial with k>2 and n=1
		
		for i in range(self.s_length):
			s = to_categorical(i, num_classes=self.s_length)
			s = np.expand_dims(s, axis=0)
			s = s.repeat(32, axis=0)

			X_drums, X_bass, X_guitar, X_strings = self.decoder.predict([s, z])
			X = dataset.postprocess(X_drums, X_bass, X_guitar, X_strings)

			X = X.reshape((32*self.bar_size, self.n_midi_programs, self.n_tracks))

			drums = pproll.Track(pianoroll=X[:, :, 0], program=0, is_drum=True, name="Drums")
			bass = pproll.Track(pianoroll=X[:, :, 1], program=33, is_drum=False, name="Bass")
			guitar = pproll.Track(pianoroll=X[:, :, 2], program=29, is_drum=False, name="Guitar")
			strings = pproll.Track(pianoroll=X[:, :, 3], program=48, is_drum=False, name="Strings")

			new_song = pproll.Multitrack(tracks=[drums, bass, guitar, strings], beat_resolution=4)
			new_song.binarize()
			new_song.assign_constant(80)
			pproll.write(new_song, os.path.join(path, prefix + "_s=" + str(i) + "_new_song.mid"))

	def save_z_latents_sweep(self, path, dataset, idx, prefix):

		# there are the indices of pitches (e.g. 68 == A4, MIDI pitch 69)
		midi_A_pitches = np.array([20, 32, 44, 56, 68, 80, 92, 104])

		# metrics for sweep evaluation
		def n_of_held_notes(X):
			# 129 (-1) is the index of held notes in the pianoroll
			amax = X.argmax(axis=-1)
			return len(amax[amax == 129])

		def n_of_silent_notes(X):
			# 128 (-2) is the index of silent notes in the pianoroll
			amax = X.argmax(axis=-1)
			return len(amax[amax == 128])

		def n_of_notes(X):
			# the first 128 pitches are actual notes
			amax = X.argmax(axis=-1)
			return len(amax[amax < 128])

		def max_pitch(X):
			amax = X.argmax(axis=-1)
			if len(amax[amax < 128]) == 0:
				return -1
			else:
				return amax[amax < 128].max()

		def min_pitch(X):
			amax = X.argmax(axis=-1)
			if len(amax[amax < 128]) == 0:
				return 128
			else:
				return amax[amax < 128].min()

		def mean_pitch(X):
			amax = X.argmax(axis=-1)
			if len(amax[amax < 128]) == 0:
				return 0
			else:
				return amax[amax < 128].mean()

		def n_of_A(X):
			amax = X.argmax(axis=-1)
			return len([x for x in amax if x in midi_A_pitches ])

		def n_of_A_sharp(X):
			amax = X.argmax(axis=-1)
			return len([x for x in amax if x in midi_A_pitches+1 ])

		def n_of_B(X):
			amax = X.argmax(axis=-1)
			return len([x for x in amax if x in midi_A_pitches+2 ])

		def n_of_C(X):
			amax = X.argmax(axis=-1)
			return len([x for x in amax if x in midi_A_pitches+3 ])

		def n_of_C_sharp(X):
			amax = X.argmax(axis=-1)
			return len([x for x in amax if x in midi_A_pitches+4 ])

		def n_of_D(X):
			amax = X.argmax(axis=-1)
			return len([x for x in amax if x in midi_A_pitches+5 ])

		def n_of_D_sharp(X):
			amax = X.argmax(axis=-1)
			return len([x for x in amax if x in midi_A_pitches+6 ])

		def n_of_E(X):
			amax = X.argmax(axis=-1)
			return len([x for x in amax if x in midi_A_pitches+7 ])

		def n_of_F(X):
			amax = X.argmax(axis=-1)
			return len([x for x in amax if x in midi_A_pitches+8 ])

		def n_of_F_sharp(X):
			amax = X.argmax(axis=-1)
			return len([x for x in amax if x in midi_A_pitches+9 ])

		def n_of_G(X):
			amax = X.argmax(axis=-1)
			return len([x for x in amax if x in midi_A_pitches+10 ])

		def n_of_G_sharp(X):
			amax = X.argmax(axis=-1)
			return len([x for x in amax if x in midi_A_pitches+11 ])

		n_repeats = 1
		pianoroll = dataset.select_pianoroll(idx)
		tmp = pianoroll.copy()
		tmp.binarize()
		tmp.assign_constant(80)

		pproll.write(tmp, os.path.join(path, "original_" + str(idx) + ".mid"))
		
		fig_tmp, _ = tmp.plot(preset="plain")
		#plt.tight_layout()
		plt.savefig(os.path.join(path, "original_" + str(idx)))
		plt.close(fig_tmp)

		X = pianoroll.get_stacked_pianoroll()

		X = np.expand_dims(X, 0)

		X, Y = dataset.preprocess(X)
		s, z = self.encoder.predict(X)

		# for each latent z, do sweep
		for i in range(self.z_length):
			z_sweep = np.copy(z)

			X_final = []

			X_drums_final = []
			X_bass_final = []
			X_guitar_final = []
			X_strings_final = []

			original = z_sweep[:, i]
			ts = np.linspace(original-self.sweep_extreme, original+self.sweep_extreme, self.sweep_granularity)
			ts_len = len(ts)

			for t in ts:
				z_sweep[:, i] = t

				# pp.pprint(z_sweep)
				# input()

				X_drums, X_bass, X_guitar, X_strings = self.decoder.predict([s, z_sweep])
				
				X_drums_final.append(X_drums)
				X_bass_final.append(X_bass)
				X_guitar_final.append(X_guitar)
				X_strings_final.append(X_strings)

				# print("X_drums:", X_drums.shape)
				# print("X_bass:", X_bass.shape)
				# print("X_guitar:", X_guitar.shape)
				# print("X_strings:", X_strings.shape)

				X_sweep = dataset.postprocess(X_drums, X_bass, X_guitar, X_strings)
				X_sweep = np.repeat(X_sweep, n_repeats, axis=0)
				X_final.append(X_sweep)

			X_final = np.concatenate(X_final, axis=0)
			X_drums_final = np.concatenate(X_drums_final, axis=0)
			X_bass_final = np.concatenate(X_bass_final, axis=0)
			X_guitar_final = np.concatenate(X_guitar_final, axis=0)
			X_strings_final = np.concatenate(X_strings_final, axis=0)

			# computing metrics for correlations
			# print("X_final:", X_final.shape)
			# print("X_drums_final:", X_drums_final.shape)
			# print("X_bass_final:", X_bass_final.shape)
			# print("X_guitar_final:", X_guitar_final.shape)
			# print("X_strings:final:", X_strings_final.shape)
			
			# after generating sweep, compute metrics

			X_final = X_final.reshape((n_repeats * ts_len * self.phrase_size, self.n_midi_pitches, self.n_tracks))
						
			# print("X_final:", X_final.shape)

			# drums metrics
			metric_drums_n_held_notes = np.array([ n_of_held_notes(chunk) for chunk in X_drums_final])
			metric_drums_n_silent_notes = np.array([ n_of_silent_notes(chunk) for chunk in X_drums_final])
			metric_drums_n_of_notes = np.array([ n_of_notes(chunk) for chunk in X_drums_final]) 
			metric_drums_max_pitch = np.array([ max_pitch(chunk) for chunk in X_drums_final])
			metric_drums_min_pitch = np.array([ min_pitch(chunk) for chunk in X_drums_final]) 
			metric_drums_mean_pitch = np.array([ mean_pitch(chunk) for chunk in X_drums_final]) 

			# bass metrics
			metric_bass_n_held_notes = np.array([ n_of_held_notes(chunk) for chunk in X_bass_final])
			metric_bass_n_silent_notes = np.array([ n_of_silent_notes(chunk) for chunk in X_bass_final])
			metric_bass_n_of_notes = np.array([ n_of_notes(chunk) for chunk in X_bass_final]) 
			metric_bass_max_pitch = np.array([ max_pitch(chunk) for chunk in X_bass_final])
			metric_bass_min_pitch = np.array([ min_pitch(chunk) for chunk in X_bass_final]) 
			metric_bass_mean_pitch = np.array([ mean_pitch(chunk) for chunk in X_bass_final]) 
			metric_bass_n_of_A = np.array([ n_of_A(chunk) for chunk in X_bass_final]) 
			metric_bass_n_of_A_sharp = np.array([ n_of_A_sharp(chunk) for chunk in X_bass_final]) 
			metric_bass_n_of_B = np.array([ n_of_B(chunk) for chunk in X_bass_final]) 
			metric_bass_n_of_C = np.array([ n_of_C(chunk) for chunk in X_bass_final]) 
			metric_bass_n_of_C_sharp = np.array([ n_of_C_sharp(chunk) for chunk in X_bass_final]) 
			metric_bass_n_of_D = np.array([ n_of_D(chunk) for chunk in X_bass_final]) 
			metric_bass_n_of_D_sharp = np.array([ n_of_D_sharp(chunk) for chunk in X_bass_final]) 
			metric_bass_n_of_E = np.array([ n_of_E(chunk) for chunk in X_bass_final]) 
			metric_bass_n_of_F = np.array([ n_of_F(chunk) for chunk in X_bass_final]) 
			metric_bass_n_of_F_sharp = np.array([ n_of_F_sharp(chunk) for chunk in X_bass_final]) 
			metric_bass_n_of_G = np.array([ n_of_G(chunk) for chunk in X_bass_final]) 
			metric_bass_n_of_G_sharp = np.array([ n_of_G_sharp(chunk) for chunk in X_bass_final]) 

			# print("number of A:", metric_bass_n_of_A)
			# print("number of A#:", metric_bass_n_of_A_sharp)
			# print("number of B:", metric_bass_n_of_B)
			# print("number of C:", metric_bass_n_of_C)
			# print("number of C#:", metric_bass_n_of_C_sharp)
			# print("number of D:", metric_bass_n_of_D)
			# print("number of D#:", metric_bass_n_of_D_sharp)
			# print("number of E:", metric_bass_n_of_E)
			# print("number of F:", metric_bass_n_of_F)
			# print("number of F#:", metric_bass_n_of_F_sharp)
			# print("number of G:", metric_bass_n_of_G)
			# print("number of G#:", metric_bass_n_of_G_sharp)			

			# # guitar metrics
			metric_guitar_n_held_notes = np.array([ n_of_held_notes(chunk) for chunk in X_guitar_final])
			metric_guitar_n_silent_notes = np.array([ n_of_silent_notes(chunk) for chunk in X_guitar_final])
			metric_guitar_n_of_notes = np.array([ n_of_notes(chunk) for chunk in X_guitar_final]) 
			metric_guitar_max_pitch = np.array([ max_pitch(chunk) for chunk in X_guitar_final])
			metric_guitar_min_pitch = np.array([ min_pitch(chunk) for chunk in X_guitar_final]) 
			metric_guitar_mean_pitch = np.array([ mean_pitch(chunk) for chunk in X_guitar_final]) 
			metric_guitar_n_of_A = np.array([ n_of_A(chunk) for chunk in X_guitar_final]) 
			metric_guitar_n_of_A_sharp = np.array([ n_of_A_sharp(chunk) for chunk in X_guitar_final]) 
			metric_guitar_n_of_B = np.array([ n_of_B(chunk) for chunk in X_guitar_final]) 
			metric_guitar_n_of_C = np.array([ n_of_C(chunk) for chunk in X_guitar_final]) 
			metric_guitar_n_of_C_sharp = np.array([ n_of_C_sharp(chunk) for chunk in X_guitar_final]) 
			metric_guitar_n_of_D = np.array([ n_of_D(chunk) for chunk in X_guitar_final]) 
			metric_guitar_n_of_D_sharp = np.array([ n_of_D_sharp(chunk) for chunk in X_guitar_final]) 
			metric_guitar_n_of_E = np.array([ n_of_E(chunk) for chunk in X_guitar_final]) 
			metric_guitar_n_of_F = np.array([ n_of_F(chunk) for chunk in X_guitar_final]) 
			metric_guitar_n_of_F_sharp = np.array([ n_of_F_sharp(chunk) for chunk in X_guitar_final]) 
			metric_guitar_n_of_G = np.array([ n_of_G(chunk) for chunk in X_guitar_final]) 
			metric_guitar_n_of_G_sharp = np.array([ n_of_G_sharp(chunk) for chunk in X_guitar_final]) 

			# # strings metrics
			metric_strings_n_held_notes = np.array([ n_of_held_notes(chunk) for chunk in X_strings_final])
			metric_strings_n_silent_notes = np.array([ n_of_silent_notes(chunk) for chunk in X_strings_final])
			metric_strings_n_of_notes = np.array([ n_of_notes(chunk) for chunk in X_strings_final]) 
			metric_strings_max_pitch = np.array([ max_pitch(chunk) for chunk in X_strings_final])
			metric_strings_min_pitch = np.array([ min_pitch(chunk) for chunk in X_strings_final]) 
			metric_strings_mean_pitch = np.array([ mean_pitch(chunk) for chunk in X_strings_final]) 
			metric_strings_n_of_A = np.array([ n_of_A(chunk) for chunk in X_strings_final]) 
			metric_strings_n_of_A_sharp = np.array([ n_of_A_sharp(chunk) for chunk in X_strings_final]) 
			metric_strings_n_of_B = np.array([ n_of_B(chunk) for chunk in X_strings_final]) 
			metric_strings_n_of_C = np.array([ n_of_C(chunk) for chunk in X_strings_final]) 
			metric_strings_n_of_C_sharp = np.array([ n_of_C_sharp(chunk) for chunk in X_strings_final]) 
			metric_strings_n_of_D = np.array([ n_of_D(chunk) for chunk in X_strings_final]) 
			metric_strings_n_of_D_sharp = np.array([ n_of_D_sharp(chunk) for chunk in X_strings_final]) 
			metric_strings_n_of_E = np.array([ n_of_E(chunk) for chunk in X_strings_final]) 
			metric_strings_n_of_F = np.array([ n_of_F(chunk) for chunk in X_strings_final]) 
			metric_strings_n_of_F_sharp = np.array([ n_of_F_sharp(chunk) for chunk in X_strings_final]) 
			metric_strings_n_of_G = np.array([ n_of_G(chunk) for chunk in X_strings_final]) 
			metric_strings_n_of_G_sharp = np.array([ n_of_G_sharp(chunk) for chunk in X_strings_final]) 

			# total metrics
			# metrics for all four track
			metric_tot_n_held_notes = metric_drums_n_held_notes + metric_bass_n_held_notes + metric_guitar_n_held_notes + metric_strings_n_held_notes
			metric_tot_n_silent_notes = metric_drums_n_silent_notes + metric_bass_n_silent_notes + metric_guitar_n_silent_notes + metric_strings_n_silent_notes
			metric_tot_n_of_notes = metric_drums_n_of_notes + metric_bass_n_of_notes + metric_guitar_n_of_notes + metric_strings_n_of_notes
			
			# metrics for all tracks except drums
			metric_tot_max_pitch = np.maximum.reduce([metric_bass_max_pitch, metric_guitar_max_pitch, metric_strings_max_pitch])
			metric_tot_min_pitch = np.minimum.reduce([metric_bass_min_pitch, metric_guitar_min_pitch, metric_strings_min_pitch])
			metric_tot_mean_pitch = np.mean([metric_bass_mean_pitch, metric_guitar_mean_pitch, metric_strings_mean_pitch], axis=0)
			
			# print(metric_bass_mean_pitch)
			# print(metric_guitar_mean_pitch)
			# print(metric_strings_mean_pitch)
			# print(metric_tot_mean_pitch)

			metric_tot_n_of_A = metric_bass_n_of_A + metric_guitar_n_of_A + metric_strings_n_of_A
			metric_tot_n_of_A_sharp = metric_bass_n_of_A_sharp + metric_guitar_n_of_A_sharp + metric_strings_n_of_A_sharp 
			metric_tot_n_of_B = metric_bass_n_of_B + metric_guitar_n_of_B + metric_strings_n_of_B 
			metric_tot_n_of_C = metric_bass_n_of_C + metric_guitar_n_of_C + metric_strings_n_of_C 
			metric_tot_n_of_C_sharp = metric_bass_n_of_C_sharp + metric_guitar_n_of_C_sharp + metric_strings_n_of_C_sharp 
			metric_tot_n_of_D = metric_bass_n_of_D + metric_guitar_n_of_D + metric_strings_n_of_D
			metric_tot_n_of_D_sharp = metric_bass_n_of_D_sharp + metric_guitar_n_of_D_sharp + metric_strings_n_of_D_sharp 
			metric_tot_n_of_E = metric_bass_n_of_E + metric_guitar_n_of_E + metric_strings_n_of_E
			metric_tot_n_of_F = metric_bass_n_of_F + metric_guitar_n_of_F + metric_strings_n_of_F
			metric_tot_n_of_F_sharp = metric_bass_n_of_F_sharp + metric_guitar_n_of_F_sharp + metric_strings_n_of_F_sharp 
			metric_tot_n_of_G = metric_bass_n_of_G + metric_guitar_n_of_G + metric_strings_n_of_G 
			metric_tot_n_of_G_sharp = metric_bass_n_of_G_sharp + metric_guitar_n_of_G_sharp + metric_strings_n_of_G_sharp

			# dict of metrics
			metrics_dict = {
				"drums_n_held_notes": pearsonr(metric_drums_n_held_notes, ts),
				"drums_n_silent_notes": pearsonr(metric_drums_n_silent_notes, ts),
				"drums_n_of_notes": pearsonr(metric_drums_n_of_notes, ts) ,
				"drums_max_pitch": pearsonr(metric_drums_max_pitch, ts),
				"drums_min_pitch":  pearsonr(metric_drums_min_pitch, ts),
				"drums_mean_pitch": pearsonr(metric_drums_mean_pitch, ts),
				"bass_n_held_notes": pearsonr(metric_bass_n_held_notes, ts),
				"bass_n_silent_notes": pearsonr(metric_bass_n_silent_notes, ts),
				"bass_n_of_notes": pearsonr(metric_bass_n_of_notes, ts),
				"bass_max_pitch": pearsonr(metric_bass_max_pitch, ts),
				"bass_min_pitch":pearsonr(metric_bass_min_pitch, ts),
				"bass_mean_pitch":pearsonr(metric_bass_mean_pitch, ts),
				# "bass_n_of_A": pearsonr(metric_bass_n_of_A, ts),
				# "bass_n_of_A_sharp":pearsonr(metric_bass_n_of_A_sharp, ts),
				# "bass_n_of_B" :pearsonr(metric_bass_n_of_B, ts),
				# "bass_n_of_C":pearsonr(metric_bass_n_of_C, ts),
				# "bass_n_of_C_sharp":pearsonr(metric_bass_n_of_C_sharp, ts),
				# "bass_n_of_D" :pearsonr(metric_bass_n_of_D, ts),
				# "bass_n_of_D_sharp":pearsonr(metric_bass_n_of_D_sharp, ts),
				# "bass_n_of_E" :pearsonr(metric_bass_n_of_E, ts),
				# "bass_n_of_F":pearsonr(metric_bass_n_of_F, ts),
				# "bass_n_of_F_sharp":pearsonr(metric_bass_n_of_F_sharp, ts),
				# "bass_n_of_G" :pearsonr(metric_bass_n_of_G, ts),
				# "bass_n_of_G_sharp" :pearsonr(metric_bass_n_of_G_sharp, ts),
				"guitar_n_held_notes":pearsonr(metric_guitar_n_held_notes, ts),
				"guitar_n_silent_notes" :pearsonr(metric_guitar_n_silent_notes, ts),
				"guitar_n_of_notes" :pearsonr(metric_guitar_n_of_notes, ts),
				"guitar_max_pitch" :pearsonr(metric_guitar_max_pitch, ts),
				"guitar_min_pitch":pearsonr(metric_guitar_min_pitch, ts),
				"guitar_mean_pitch":pearsonr(metric_guitar_mean_pitch, ts),
				# "guitar_n_of_A" :pearsonr(metric_guitar_n_of_A, ts),
				# "guitar_n_of_A_sharp":pearsonr(metric_guitar_n_of_A_sharp, ts),
				# "guitar_n_of_B" :pearsonr(metric_guitar_n_of_B, ts),
				# "guitar_n_of_C":pearsonr(metric_guitar_n_of_C, ts),
				# "guitar_n_of_C_sharp":pearsonr(metric_guitar_n_of_C_sharp, ts),
				# "guitar_n_of_D" :pearsonr(metric_guitar_n_of_D, ts),
				# "guitar_n_of_D_sharp":pearsonr(metric_guitar_n_of_D_sharp, ts),
				# "guitar_n_of_E" :pearsonr(metric_guitar_n_of_E, ts),
				# "guitar_n_of_F":pearsonr(metric_guitar_n_of_F, ts),
				# "guitar_n_of_F_sharp":pearsonr(metric_guitar_n_of_F_sharp, ts),
				# "guitar_n_of_G" :pearsonr(metric_guitar_n_of_G, ts),
				# "guitar_n_of_G_sharp" :pearsonr(metric_guitar_n_of_G_sharp, ts),
				"strings_n_held_notes":pearsonr(metric_strings_n_held_notes, ts),
				"strings_n_silent_notes" :pearsonr(metric_strings_n_silent_notes, ts),
				"strings_n_of_notes" :pearsonr(metric_strings_n_of_notes, ts),
				"strings_max_pitch" :pearsonr(metric_strings_max_pitch, ts),
				"strings_min_pitch":pearsonr(metric_strings_min_pitch, ts),
				"strings_mean_pitch":pearsonr(metric_strings_mean_pitch, ts),
				# "strings_n_of_A" :pearsonr(metric_strings_n_of_A, ts),
				# "strings_n_of_A_sharp":pearsonr(metric_strings_n_of_A_sharp, ts),
				# "strings_n_of_B" :pearsonr(metric_strings_n_of_B, ts),
				# "strings_n_of_C":pearsonr(metric_strings_n_of_C, ts),
				# "strings_n_of_C_sharp":pearsonr(metric_strings_n_of_C_sharp, ts),
				# "strings_n_of_D" :pearsonr(metric_strings_n_of_D, ts),
				# "strings_n_of_D_sharp":pearsonr(metric_strings_n_of_D_sharp, ts),
				# "strings_n_of_E" :pearsonr(metric_strings_n_of_E, ts),
				# "strings_n_of_F":pearsonr(metric_strings_n_of_F, ts),
				# "strings_n_of_F_sharp":pearsonr(metric_strings_n_of_F_sharp, ts),
				# "strings_n_of_G" :pearsonr(metric_strings_n_of_G, ts),
				# "strings_n_of_G_sharp" :pearsonr(metric_strings_n_of_G_sharp, ts),
				"tot_n_held_notes":pearsonr(metric_tot_n_held_notes, ts),
				"tot_n_silent_notes" :pearsonr(metric_tot_n_silent_notes, ts),
				"tot_n_of_notes" :pearsonr(metric_tot_n_of_notes, ts),
				"tot_max_pitch" :pearsonr(metric_tot_max_pitch, ts),
				"tot_min_pitch":pearsonr(metric_tot_min_pitch, ts),
				"tot_mean_pitch":pearsonr(metric_tot_mean_pitch, ts)#,
				# "tot_n_of_A" :pearsonr(metric_tot_n_of_A, ts),
				# "tot_n_of_A_sharp":pearsonr(metric_tot_n_of_A_sharp, ts),
				# "tot_n_of_B" :pearsonr(metric_tot_n_of_B, ts),
				# "tot_n_of_C":pearsonr(metric_tot_n_of_C, ts),
				# "tot_n_of_C_sharp":pearsonr(metric_tot_n_of_C_sharp, ts),
				# "tot_n_of_D" :pearsonr(metric_tot_n_of_D, ts),
				# "tot_n_of_D_sharp":pearsonr(metric_tot_n_of_D_sharp, ts),
				# "tot_n_of_E" :pearsonr(metric_tot_n_of_E, ts),
				# "tot_n_of_F":pearsonr(metric_tot_n_of_F, ts),
				# "tot_n_of_F_sharp":pearsonr(metric_tot_n_of_F_sharp, ts),
				# "tot_n_of_G" : pearsonr(metric_tot_n_of_G, ts),
				# "tot_n_of_G_sharp": pearsonr(metric_tot_n_of_G_sharp, ts)
			}

			# save correlations to a file
			# TODO aggiungere info su qualipianoroll è salvato
			with open(os.path.join(path, prefix + "_" + str(idx) + "_correlations_z_factor_"+ str(i) +".txt"), "w") as fp:
				pprint.pprint(metrics_dict, stream=fp)

			# matrix of metrics
			metrics = np.array([
				ts,
				metric_drums_n_held_notes,
				metric_drums_n_silent_notes,
				metric_drums_n_of_notes,
				metric_drums_max_pitch,
				metric_drums_min_pitch ,
				metric_drums_mean_pitch,
				metric_bass_n_held_notes,
				metric_bass_n_silent_notes ,
				metric_bass_n_of_notes ,
				metric_bass_max_pitch ,
				metric_bass_min_pitch,
				metric_bass_mean_pitch,
				# metric_bass_n_of_A ,
				# metric_bass_n_of_A_sharp,
				# metric_bass_n_of_B ,
				# metric_bass_n_of_C,
				# metric_bass_n_of_C_sharp,
				# metric_bass_n_of_D ,
				# metric_bass_n_of_D_sharp,
				# metric_bass_n_of_E ,
				# metric_bass_n_of_F,
				# metric_bass_n_of_F_sharp,
				# metric_bass_n_of_G ,
				# metric_bass_n_of_G_sharp ,
				metric_guitar_n_held_notes,
				metric_guitar_n_silent_notes ,
				metric_guitar_n_of_notes ,
				metric_guitar_max_pitch ,
				metric_guitar_min_pitch,
				metric_guitar_mean_pitch,
				# metric_guitar_n_of_A ,
				# metric_guitar_n_of_A_sharp,
				# metric_guitar_n_of_B ,
				# metric_guitar_n_of_C,
				# metric_guitar_n_of_C_sharp,
				# metric_guitar_n_of_D ,
				# metric_guitar_n_of_D_sharp,
				# metric_guitar_n_of_E ,
				# metric_guitar_n_of_F,
				# metric_guitar_n_of_F_sharp,
				# metric_guitar_n_of_G ,
				# metric_guitar_n_of_G_sharp ,
				metric_strings_n_held_notes,
				metric_strings_n_silent_notes ,
				metric_strings_n_of_notes ,
				metric_strings_max_pitch ,
				metric_strings_min_pitch,
				metric_strings_mean_pitch,
				# metric_strings_n_of_A ,
				# metric_strings_n_of_A_sharp,
				# metric_strings_n_of_B ,
				# metric_strings_n_of_C,
				# metric_strings_n_of_C_sharp,
				# metric_strings_n_of_D ,
				# metric_strings_n_of_D_sharp,
				# metric_strings_n_of_E ,
				# metric_strings_n_of_F,
				# metric_strings_n_of_F_sharp,
				# metric_strings_n_of_G ,
				# metric_strings_n_of_G_sharp ,
				metric_tot_n_held_notes,
				metric_tot_n_silent_notes ,
				metric_tot_n_of_notes ,
				metric_tot_max_pitch ,
				metric_tot_min_pitch,
				metric_tot_mean_pitch#,
				# metric_tot_n_of_A ,
				# metric_tot_n_of_A_sharp,
				# metric_tot_n_of_B ,
				# metric_tot_n_of_C,
				# metric_tot_n_of_C_sharp,
				# metric_tot_n_of_D ,
				# metric_tot_n_of_D_sharp,
				# metric_tot_n_of_E ,
				# metric_tot_n_of_F,
				# metric_tot_n_of_F_sharp,
				# metric_tot_n_of_G ,
				# metric_tot_n_of_G_sharp
			])
			correlations = np.corrcoef(metrics)
			
			plt.imshow(correlations, interpolation='none', cmap=plt.cm.bwr)
			plt.colorbar()

			ticks = ["z"] + list(metrics_dict.keys())
			nticks = range(len(ticks))

			plt.yticks(nticks, ticks, rotation=0)
			plt.xticks(nticks, ticks, rotation=90)

			plt.savefig(os.path.join(path, prefix + "_" + str(idx) + "_correlations_matrix_z_factor_"+str(i)))
			plt.clf()
			plt.close(plt.gcf())

			# saving sweeped song
			pianoroll_sweep = pianoroll.copy()
			for t, track in enumerate(pianoroll_sweep.tracks):
				track.pianoroll = X_final[:, :, t]

			pianoroll_sweep.binarize()
			pianoroll_sweep.assign_constant(80)
			pproll.write(pianoroll_sweep, os.path.join(path, prefix + "_"+str(ts_len)+"_idx_"+str(idx)+"_z_factor_"+str(i)+".mid"))
			fig_sweep, _ = pianoroll_sweep.plot(preset="plain")
			#plt.tight_layout()
			plt.savefig(os.path.join(path, prefix + "_"+str(ts_len)+"_idx_"+str(idx)+"_z_factor_"+str(i)))
			plt.close(fig_sweep)



	def style_transfer(self, path, dataset, idx, prefix):
		metadata, multitrack = dataset.select_song(idx, metadata=True)
		print("Style transfer of", metadata["artist"] + " - " + metadata["title"])

		metadata["artist"] = metadata["artist"].replace("/", "-")
		metadata["title"] = metadata["title"].replace("/", "-")

		guitar_tracks, bass_tracks, drums_tracks, string_tracks = dataset.get_guitar_bass_drums(multitrack)

		# ----------------------
		# Store original track
		#-----------------------
		pproll.write(multitrack, os.path.join(path, "original_" + metadata["artist"] + " - "+ metadata["title"] + ".mid"))
		fig_mt, _ = multitrack.plot(preset="plain")
		#plt.tight_layout()
		plt.savefig(os.path.join(path, "original_" + metadata["artist"] + " - "+ metadata["title"]))
		plt.close(fig_mt)

		# take all possible combination of guitar, bass and drums
		i = 0 
		for guitar_track in guitar_tracks:
			for bass_track in bass_tracks:
				for drums_track in drums_tracks:
					i += 1
					# hand-made selection of chosen song's track combination
					if idx == 1190 and (i !=  1):
						continue
					elif idx == 210 and (i !=  1):
						continue
					elif idx == 2454 and (i !=  1):
						continue
					elif idx == 340 and (i !=  1):
						continue
					elif idx == 2932 and (i !=  1):
						continue
					elif idx == 2853 and (i !=  1):
						continue
					elif idx == 1979 and (i !=  1):
						continue
					elif idx == 657 and (i !=  1):
						continue
					elif idx == 2149 and (i !=  1):
						continue
					elif idx == 2112 and (i !=  1):
						continue
					elif idx == 1882 and (i !=  1):
						continue
					# elif idx == 1119 and (i !=  1):
					# 	continue

					# # juds priest breaking the law
					# self.style_transfer(paths["style_transfers"], dataset, 210, str(pbc_tr))
					
					# # megadeth - angry again
					# self.style_transfer(paths["style_transfers"], dataset, 340, str(pbc_tr))
					
					# # abba - hasta manana
					# self.style_transfer(paths["style_transfers"], dataset, 1190, str(pbc_tr))
					
					# # queen - fat bottomed girls
					# self.style_transfer(paths["style_transfers"], dataset, 2932, str(pbc_tr))
					
					# # iron maiden - halloweed be thy name
					# self.style_transfer(paths["style_transfers"], dataset, 2890, str(pbc_tr))
					
					# # Ricky Valance - Fly Me To The Moon
					# self.style_transfer(paths["style_transfers"], dataset, 2853, str(pbc_tr))

					# # 2454 -> Madness - One Step Beyond
					# self.style_transfer(paths["style_transfers"], dataset, 2454, str(pbc_tr))

					# # 2371 -> Andrew Gold - Lady Madonna
					# self.style_transfer(paths["style_transfers"], dataset, 2371, str(pbc_tr))
					
					# # 2148 -> The Police - Walking On The Moon
					# self.style_transfer(paths["style_transfers"], dataset, 2148, str(pbc_tr))
					
					# # 1979 -> The Police - Spirits In The Material World
					# self.style_transfer(paths["style_transfers"], dataset, 1979, str(pbc_tr))
					
					# # 2059 -> Bob Marley & The Wailers - Could You Be Loved
					# self.style_transfer(paths["style_transfers"], dataset, 2059, str(pbc_tr))
					
					# # 1667 -> The Mamas & The Papas - California Dreamin'
					# self.style_transfer(paths["style_transfers"], dataset, 1667, str(pbc_tr))
					
					# # 1880 -> Robert Palmer - Bad Case Of Loving You (Doctor_ Doctor)
					# self.style_transfer(paths["style_transfers"], dataset, 1880, str(pbc_tr))
					
					# # 657 -> Elton John - Spirit In The Sky
					# self.style_transfer(paths["style_transfers"], dataset, 657, str(pbc_tr))
					
					# # 2112 -> ZZ Top - Just Got Paid
					# self.style_transfer(paths["style_transfers"], dataset, 2112, str(pbc_tr))
					
					# # 2947 -> Pooh - In Diretta Nel Vento
					# self.style_transfer(paths["style_transfers"], dataset, 2947, str(pbc_tr))
					
					# # 252 -> Luciano & Silvana Blue - Non voglio mica la luna
					# self.style_transfer(paths["style_transfers"], dataset, 252, str(pbc_tr))

					current_tracks = [drums_track, bass_track, guitar_track, -1]
					names = ["Drums", "Bass", "Guitar", "Strings"]

					# create temporary song with only that tracks
					song = pproll.Multitrack()
					song.remove_empty_tracks()
					
					for j, current_track in enumerate(current_tracks):
						song.append_track(
							pianoroll=multitrack.tracks[current_track].pianoroll,
							program=multitrack.tracks[current_track].program,
							is_drum=multitrack.tracks[current_track].is_drum,
							name=names[j]
						)
					song.name = metadata["artist"] + " - "+ metadata["title"]
					song.beat_resolution = multitrack.beat_resolution
					song.tempo = multitrack.tempo

					# --------------
					# ready to store original (preprocessed) and autoencoded song
					# --------------
					# replacing original tracks with preprocessed
					song.binarize()
					song.assign_constant(1)
					X = song.get_stacked_pianoroll()

					#using n_midi_pitches instead of n_cropped_notes because it's actual song pianoroll
					X = np.reshape(X, (-1, self.phrase_size, self.n_midi_pitches, self.n_tracks))
					X, Y = dataset.preprocess(X)
					Y = dataset.postprocess(Y[:, :, :, 0], Y[:, :, :, 1], Y[:, :, :, 2], Y[:, :, :, 3])
					
					# for i, j in itertools.product(range(Y.shape[0]), range(Y.shape[1])):#, range(X.shape[3])):
					# 	print(Y[i, j, :, 1])
					# 	print(len(Y[i, j, :, 1]))
					# 	input()

					#print("2 Y shape:", Y.shape)
					Y = np.reshape(Y, (-1, self.n_midi_pitches, self.n_tracks))
					for t, track in enumerate(song.tracks):
						track.pianoroll = Y[:, :, t]	
					
					song.binarize()
					song.assign_constant(80)
					fig_song, _ = song.plot(preset="plain")
					#plt.tight_layout()
					plt.savefig(os.path.join(path, "original_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(i)))					
					plt.close(fig_song)
					pproll.write(song, os.path.join(path, "original_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(i) + ".mid"))

					m50 = song.copy()
					
					s, z, _ = self.autoencode(X, dataset)
					
					# plot latent s factor
					plt.figure()
					plt.imshow(s, interpolation='none', cmap=plt.cm.bwr)
					plt.colorbar()
					plt.title("s latent factors for " + metadata["artist"] + " - "+ metadata["title"])
					plt.savefig(os.path.join(path, prefix + "_s_latent_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(i)))
					plt.clf()
					plt.close(plt.gcf())

					with open(os.path.join(self.dataset_path, "genre_counter.json"), "r") as counter_fp:
						genre_counter = json.load(counter_fp)
						
						# plotting detected style
						with open(os.path.join(path, prefix + "_s_latent_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(i) + ".txt"), "w") as dest_fp:
							for k in range(self.s_length):
								dest_fp.write(str(k) + " : " + genre_counter[k][0] + " = " + str(s[:, k].mean()) + "\n")

						# setting different s factors
						for k in range(self.s_length):
							genre_name = genre_counter[k][0]
							s1 = np.zeros(s.shape)
							s1[:, k] = 1
							X_drums, X_bass, X_guitar, X_strings = self.decoder.predict([s1, z])
							
							X_rec = dataset.postprocess(X_drums, X_bass, X_guitar, X_strings)
							X_rec = np.reshape(X_rec, (-1, self.n_midi_pitches, self.n_tracks))

							# plt.figure(figsize=(50, 5))
							# plt.imshow(z, interpolation='none', cmap=plt.cm.bwr)
							# plt.colorbar(orientation="horizontal")
							# plt.title("z latent factors for " + metadata["artist"] + " - "+ metadata["title"])
							# plt.savefig(os.path.join(path,  prefix + "_z_latent_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(i)))
							# plt.clf()
							# plt.close(plt.gcf())
							# plt.figure()
							# plt.imshow(s, interpolation='none', cmap=plt.cm.bwr)
							# plt.colorbar()
							# plt.title("s latent factors for " + metadata["artist"] + " - "+ metadata["title"])
							# plt.savefig(os.path.join(path, prefix + "_s_latent_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(i)))
							# plt.clf()
							# plt.close(plt.gcf())
							
							# replacing original tracks with style transfer tracks
							for t, track in enumerate(m50.tracks):
								track.pianoroll = X_rec[:, :, t]

							m50.assign_constant(80)
							pproll.write(m50, os.path.join(path, prefix + "_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(k) +  "_" + genre_name +".mid"))
							fig_m50, _ = m50.plot(preset="plain")
							#plt.tight_layout()
							plt.savefig(os.path.join(path, prefix + "_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(k) + "_" + genre_name))
							plt.close(fig_m50)
							

	def save_interpolation(self, path, dataset, idx1, idx2, prefix):
		n_repeats = 2
		pianoroll1 = dataset.select_pianoroll(idx1)
		pianoroll2 = dataset.select_pianoroll(idx2)

		tmp1 = pianoroll1.copy()
		tmp2 = pianoroll2.copy()
		tmp1.binarize()
		tmp2.binarize()
		tmp1.assign_constant(80)
		tmp2.assign_constant(80)

		# tmp1proll = tmp1.get_stacked_pianoroll()
		# tmp2proll = tmp2.get_stacked_pianoroll()
		# for t, track in enumerate(tmp1.tracks):
		# 	track.pianoroll = np.repeat(tmp1proll[:, :, t], 4, axis=0)
		
		# # print("tmp1proll: ", tmp1proll.shape)
		# # print("tmp1proll single track: ", tmp1proll[:, :, 0].shape)
		# # print("tmp1proll single track repeat: ", np.repeat(tmp1proll[:, :, 0], 4, axis=0).shape)

		# # exit(0)
		# for t, track in enumerate(tmp2.tracks):
		# 	track.pianoroll = np.repeat(tmp2proll[:, :, t], 4, axis=0)

		pproll.write(tmp1, os.path.join(path, "original_" + str(idx1) + ".mid"))
		pproll.write(tmp2, os.path.join(path, "original_" + str(idx2) + ".mid"))
		
		fig_tmp1, _ = tmp1.plot(preset="plain")
		#plt.tight_layout()
		plt.savefig(os.path.join(path, "original_" + str(idx1)))
		plt.close(fig_tmp1)

		fig_tmp2, _ = tmp2.plot(preset="plain")
		#plt.tight_layout()
		plt.savefig(os.path.join(path, "original_" + str(idx2)))
		plt.close(fig_tmp2)
		#del tmp1
		#del tmp2

		X1 = pianoroll1.get_stacked_pianoroll()
		X2 = pianoroll2.get_stacked_pianoroll()

		X1 = np.expand_dims(X1, 0)
		X2 = np.expand_dims(X2, 0)

		X1, Y1 = dataset.preprocess(X1)
		X2, Y2 = dataset.preprocess(X2)

		# TODO store also autoencoded version of start and end bar
		s1, z1 = self.encoder.predict(X1)
		s2, z2 = self.encoder.predict(X2)

		def linear_interpolation(p0, p1, ts):
			for t in ts:
				yield p0 * (1.0-t) + p1 * t

		for ts in [np.linspace(0, 1, 5), np.linspace(0, 1, 7), np.linspace(0, 1, 9)]:
			#k = 0
			for k, s_int in enumerate([s1, s2]):	
				ts_len = len(ts)
				X_final = [] # np.zeros((5, self.bar_size, self.n_cropped_notes, self.n_tracks))
				for z_int in linear_interpolation(z1, z2, ts):			
					X_drums, X_bass, X_guitar, X_strings = self.decoder.predict([s_int, z_int])
					X_int = dataset.postprocess(X_drums, X_bass, X_guitar, X_strings)
					
					# print("X int.shape", X_int.shape)

					X_int = np.repeat(X_int, n_repeats, axis=0)

					# print("X int.shape", X_int.shape)

					X_final.append(X_int)			

				X_final = np.concatenate(X_final, axis=0)

				# print("X final.shape", X_final.shape)
				
				# #X_final.repeat(X_final, 2, axis=)

				# print("X final.shape", X_final.shape)				

				# exit(0)

				X_final = X_final.reshape((n_repeats * ts_len * self.phrase_size, self.n_midi_pitches, self.n_tracks))
				
				# saving interpolated song
				pianoroll_int = pianoroll1.copy()
				for t, track in enumerate(pianoroll_int.tracks):
					track.pianoroll = X_final[:, :, t]

				pianoroll_int.binarize()
				pianoroll_int.assign_constant(80)
				pproll.write(pianoroll_int, os.path.join(path, prefix + "_"+str(ts_len)+"_linear_"+str(idx1)+"_to_"+str(idx2)+"_v"+str(k)+".mid"))
				fig_int, _ = pianoroll_int.plot(preset="plain")
				#plt.tight_layout()
				plt.savefig(os.path.join(path, prefix + "_"+str(ts_len)+"_linear_"+str(idx1)+"_to_"+str(idx2)+"_v"+str(k)))
				plt.close(fig_int)


	def save_autoencoded_song(self, path, dataset, idx, prefix):
		metadata, multitrack = dataset.select_song(idx, metadata=True)
		print("Autoencoding", metadata["artist"] + " - " + metadata["title"])

		metadata["artist"] = metadata["artist"].replace("/", "-")
		metadata["title"] = metadata["title"].replace("/", "-")

		guitar_tracks, bass_tracks, drums_tracks, string_tracks = dataset.get_guitar_bass_drums(multitrack)

		# ----------------------
		# Store original track
		#-----------------------
		pproll.write(multitrack, os.path.join(path, "original_" + metadata["artist"] + " - "+ metadata["title"] + ".mid"))
		fig_mt, _ = multitrack.plot(preset="plain")
		#plt.tight_layout()
		plt.savefig(os.path.join(path, "original_" + metadata["artist"] + " - "+ metadata["title"]))
		plt.close(fig_mt)

		# take all possible combination of guitar, bass and drums
		i = 0 
		for guitar_track in guitar_tracks:
			for bass_track in bass_tracks:
				for drums_track in drums_tracks:
					i += 1
					# hand-made selection of chosen song's track combination
					if idx == 53 and (i !=  1):
						continue
					elif idx == 210 and (i !=  1):
						continue
					elif idx == 257 and (i !=  1):
						continue
					elif idx == 318 and (i !=  1):
						continue
					elif idx == 406 and (i !=  1):
						continue
					elif idx == 577 and (i !=  1):
						continue
					elif idx == 815 and (i !=  1):
						continue
					elif idx == 1993 and (i !=  1):
						continue
					elif idx == 946 and (i !=  3):
						continue
					elif idx == 992 and (i !=  3):
						continue
					elif idx == 1109 and (i !=  7):
						continue
					elif idx == 1119 and (i !=  1):
						continue

					current_tracks = [drums_track, bass_track, guitar_track, -1]
					names = ["Drums", "Bass", "Guitar", "Strings"]

					# create temporary song with only that tracks
					song = pproll.Multitrack()
					song.remove_empty_tracks()
					
					for j, current_track in enumerate(current_tracks):
						song.append_track(
							pianoroll=multitrack.tracks[current_track].pianoroll,
							program=multitrack.tracks[current_track].program,
							is_drum=multitrack.tracks[current_track].is_drum,
							name=names[j]
						)
					song.name = metadata["artist"] + " - "+ metadata["title"]
					song.beat_resolution = multitrack.beat_resolution
					song.tempo = multitrack.tempo

					# --------------
					# ready to store original (preprocessed) and autoencoded song
					# --------------
					# replacing original tracks with preprocessed
					song.binarize()
					song.assign_constant(1)
					X = song.get_stacked_pianoroll()

					#using n_midi_pitches instead of n_cropped_notes because it's actual song pianoroll
					X = np.reshape(X, (-1, self.phrase_size, self.n_midi_pitches, self.n_tracks))
					X, Y = dataset.preprocess(X)
					Y = dataset.postprocess(Y[:, :, :, 0], Y[:, :, :, 1], Y[:, :, :, 2], Y[:, :, :, 3])
					
					# for i, j in itertools.product(range(Y.shape[0]), range(Y.shape[1])):#, range(X.shape[3])):
					# 	print(Y[i, j, :, 1])
					# 	print(len(Y[i, j, :, 1]))
					# 	input()

					#print("2 Y shape:", Y.shape)
					Y = np.reshape(Y, (-1, self.n_midi_pitches, self.n_tracks))
					for t, track in enumerate(song.tracks):
						track.pianoroll = Y[:, :, t]	
					
					song.binarize()
					song.assign_constant(80)
					fig_song, _ = song.plot(preset="plain")
					#plt.tight_layout()
					plt.savefig(os.path.join(path, "original_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(i)))					
					plt.close(fig_song)
					pproll.write(song, os.path.join(path, "original_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(i) + ".mid"))

					m50 = song.copy()
					
					s, z, X_rec = self.autoencode(X, dataset)
					X_rec = np.reshape(X_rec, (-1, self.n_midi_pitches, self.n_tracks))

					# plot histogram of latent (to check they follow the desired distribution)
					#plt.figure()
					z_flat = z.flatten()
					plt.hist(z_flat, bins='auto')
					plt.title("Z latents distribution")
					plt.savefig(os.path.join(path,  prefix + "_z_histogram_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(i)))
					plt.close(plt.gcf())

					#plt.figure()
					s_flat = s.flatten()
					plt.hist(s_flat, bins=self.s_length)
					plt.title("S latents distribution")
					plt.savefig(os.path.join(path,  prefix + "_s_histogram_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(i)))
					plt.close(plt.gcf())

					plt.figure(figsize=(50, 5))
					plt.imshow(z, interpolation='none', cmap=plt.cm.bwr)
					plt.colorbar(orientation="horizontal")
					plt.title("z latent factors for " + metadata["artist"] + " - "+ metadata["title"])
					plt.savefig(os.path.join(path,  prefix + "_z_latent_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(i)))
					plt.clf()
					plt.close(plt.gcf())
					plt.figure()
					plt.imshow(s, interpolation='none', cmap=plt.cm.bwr)
					plt.colorbar()
					plt.title("s latent factors for " + metadata["artist"] + " - "+ metadata["title"])
					plt.savefig(os.path.join(path, prefix + "_s_latent_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(i)))
					plt.clf()
					plt.close(plt.gcf())
					# replacing original tracks with autoencoded tracks
					for t, track in enumerate(m50.tracks):
						track.pianoroll = X_rec[:, :, t]

					m50.assign_constant(80)
					pproll.write(m50, os.path.join(path, prefix + "_autoencoded_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(i) +  ".mid"))
					fig_m50, _ = m50.plot(preset="plain")
					#plt.tight_layout()
					plt.savefig(os.path.join(path, prefix + "_autoencoded_" + metadata["artist"] + " - "+ metadata["title"] + "_" + str(i)))
					plt.close(fig_m50)

	



	# TODO mettere a posto parte dei checkpoint
	def save_checkpoint(self, path, epoch):
		self.adversarial_autoencoder.save(os.path.join(path, str(epoch) + "aae.h5"))

	def load_checkpoint(self, path, epoch):
		self.adversarial_autoencoder = load_model(
			os.path.join(path, str(epoch) + "aae.h5"),
			custom_objects={
				"RandomWeightedAverage": RandomWeightedAverage,
				"wasserstein_loss": self.wasserstein_loss,
				"gradient_penalty_s": self.s_gp_loss,
				"gradient_penalty_z": self.z_gp_loss,
				"output": self.output
			},
		)
		self.z_regularisation_phase = self.adversarial_autoencoder.z_regularisation_phase
		self.s_regularisation_phase = self.adversarial_autoencoder.s_regularisation_phase
		self.reconstruction_phase = self.adversarial_autoencoder.reconstruction_phase
		self.encoder = self.reconstruction_phase.encoder
		self.decoder = self.reconstruction_phase.decoder
		self.s_discriminator = self.s_regularisation_phase.s_discriminator
		self.z_discriminator = self.z_regularisation_phase.z_discriminator
