import config
import numpy as np
import time
import os
import pretty_midi as pm
import math
import random
import json
import pickle
import progressbar
from sklearn.model_selection import train_test_split
import tables
import matplotlib.pyplot as plt
import pypianoroll as pproll
import pprint
import itertools
from keras.utils import to_categorical
from collections import Counter

pp = pprint.PrettyPrinter(indent=4)

class MidiDataset():
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

	def select_batch(self, idx):
		X = np.load(os.path.join(self.dataset_path, "batches", "X", str(idx) + ".npy"))
		Y = np.load(os.path.join(self.dataset_path, "batches", "Y", str(idx) + ".npy"))
		label = np.load(os.path.join(self.dataset_path, "batches", "labels", str(idx) + ".npy"))
		return X, Y, label

	def select_song(self, idx, metadata=True):
		_metadata_ = None

		multitrack = pproll.load(os.path.join(self.dataset_path + "songs/" + str(idx) + ".npz"))

		if metadata:
			_metadata_ = self.retrieve_metadata(os.path.join(self.dataset_path, "metadata/", str(idx) + ".json"))

		return _metadata_, multitrack

	def select_pianoroll(self, idx):
		pianoroll = pproll.load(os.path.join(self.dataset_path + "pianorolls/" + str(idx) + ".npz"))
		return pianoroll

	def get_programs(self, song):	
		programs = np.array([ track.program for track in song.tracks ])
		return programs
	
	# I shape: (n_midi_programs, n_tracks)
	def programs_to_instrument_matrix(self, programs):    
		assert(len(programs) == self.n_tracks)
		
		I = np.zeros((self.n_midi_programs, self.n_tracks))
		for i, program in enumerate(programs):
			I[program, i] = 1

		return I

	# I shape: (n_midi_programs, n_tracks)
	def instrument_matrix_to_programs(self, I):
		assert(I.shape[1] == self.n_tracks)
		assert(I.shape[0] == self.n_midi_programs)

		programs = [ np.argmax(I[:, i]) for i in range(I.shape[1]) ]

		return np.array(programs)
	
	def retrieve_metadata(self, path):
		with open(path, "r") as fp:
			metadata = json.load(fp)
		return metadata

	def retrieve_pianoroll_metadata(self, meta_link, idx):
		if not isinstance(idx, str):
			raise TypeError("idx must be a string")
		song_id = meta_link[idx]
		return self.retrieve_metadata(os.path.join(self.dataset_path, "metadata/", str(song_id) + ".json"))

	def retrieve_instrument_matrix(self, path):	
		I = np.load(path)
		return I

	def generate_batches(self, path, filenames, batch_size):
		print("Generating batches from data...")
		dataset_len = len(filenames)
		# shuffle samples 
		random.shuffle(filenames)
		
		# discard filenames
		remainder = dataset_len % batch_size
		dataset = np.array(filenames[:-remainder])
		dataset_len = dataset.shape[0]

		assert(dataset_len % batch_size == 0)
		dataset = dataset.reshape((-1, batch_size))
		n_of_batches = dataset.shape[0]

		for i in range(n_of_batches):
			source = dataset[i, :]
			dest = []
			for sample in source:
				multitrack = pproll.load(os.path.join(path, sample))
				proll = multitrack.get_stacked_pianoroll()
				dest.append(proll)

			dest = np.array(dest)
			yield dest

	# warning: tends to use a lot of storage (disk) space
	def create_batches(self, batch_size=128):
		print("Building batches from data...")

		batch_path = os.path.join(self.dataset_path, "batches/")
		if not os.path.exists(batch_path):
			os.makedirs(os.path.join(batch_path, "X"))
			os.makedirs(os.path.join(batch_path, "Y"))
			os.makedirs(os.path.join(batch_path, "labels"))

		pianorolls_path = os.path.join(self.dataset_path, "pianorolls/")
		metadata_path = os.path.join(self.dataset_path, "metadata/")

		_, _, files = next(os.walk(pianorolls_path))
		
		dataset_len = len(files)

		random.shuffle(files)
		remainder = dataset_len % batch_size
		dataset = np.array(files[:-remainder])
		dataset_len = dataset.shape[0]

		print("dataset_length:", dataset_len)
		print("batch_size:", batch_size)
		print("number of batches:", dataset_len // batch_size)
		print("remainder:", remainder)

		assert(dataset_len % batch_size == 0)
		dataset = dataset.reshape((-1, batch_size))
		n_of_batches = dataset.shape[0]

		# store each batch in a file toghether
		bar = progressbar.ProgressBar(max_value=n_of_batches)

		meta_link = json.load(open(os.path.join(self.dataset_path, "meta_link.json")))
		for i in range(n_of_batches):
			bar.update(i)
			source = dataset[i, :]
			dest = []
			labels = []
			# for each pianoroll, store it and the corresponding labels
			for sample in source:
				multitrack = pproll.load(os.path.join(pianorolls_path, sample))
				proll = multitrack.get_stacked_pianoroll()
				dest.append(proll)

				# retrieve corresponding s factors
				sample_id = sample.split(".")[0]
				song_id = meta_link[sample_id]
				label = np.load(os.path.join(self.dataset_path, "labels", str(song_id) + ".npy"))
				labels.append(label)

			dest = np.array(dest)
			labels = np.array(labels)
			# preprocess batch, get X and Y
			X, Y = self.preprocess(dest)
			# store everything
			np.save(os.path.join(batch_path, "X", str(i) + ".npy"), X)
			np.save(os.path.join(batch_path, "Y", str(i) + ".npy"), Y)
			np.save(os.path.join(batch_path, "labels", str(i) + ".npy"), labels)

	def preprocess(self, X):
		# if silent timestep (all 0), then set silent note to 1, else set
		# silent note to 0
		def pad_with(vector, pad_width, iaxis, kwargs):			
			# if no padding, skip directly
			if pad_width[0] == 0 and pad_width[1] == 0:
				return vector
			else:

				if all(vector[pad_width[0]:-pad_width[1]] == 0):
					
					pad_value = 1
				else:
					pad_value = 0

				vector[:pad_width[0]] = pad_value
				vector[-pad_width[1]:] = pad_value


		# adding silent note
		X = np.pad(X, ((0, 0), (0, 0), (0, 2), (0, 0)), mode=pad_with)

		# converting to categorical (keep only one note played at a time)
		tracks = []
		for t in range(self.n_tracks):
			X_t = X[:, :, :, t]
			X_t = to_categorical(X_t.argmax(2), num_classes=self.n_cropped_notes)
			X_t = np.expand_dims(X_t, axis=-1)

			tracks.append(X_t)
		
		X = np.concatenate(tracks, axis=-1)
		
		# adding held note
		for sample in range(X.shape[0]):
			for ts in range(1, X.shape[1]):
				for track in range(X.shape[3]):
					# check for equality, except for the hold note position (the last position)
					if np.array_equal(X[sample, ts, :-1, track], X[sample, ts-1, :-1, track]):
						X[sample, ts, -1, track] = 1

		#just zero the pianoroll where there is a held note
		for sample in range(X.shape[0]):
			for ts in range(1, X.shape[1]):
				for track in range(X.shape[3]):
					if X[sample, ts, -1, track] == 1:
						X[sample, ts, :-1, track] = 0

		# finally, use [0, 1] interval for ground truth Y and [-1, 1] interval for input/teacher forcing X
		Y = X.copy()
		X[X == 1] = 1
		X[X == 0] = -1

		return X, Y


	def postprocess(self, X_drums, X_bass, X_guitar, X_strings):
		#putting tracks back toghether
		batch_size = X_drums.shape[0]
		n_timesteps = X_drums.shape[1]
		
		# converting softmax outputs to categorical
		tracks = []
		for track in [X_drums, X_bass, X_guitar, X_strings]:
			track = to_categorical(track.argmax(2), num_classes=self.n_cropped_notes)
			track = np.expand_dims(track, axis=-1)
			tracks.append(track)
		
		X = np.concatenate(tracks, axis=-1)

		# copying previous timestep if held note is on
		for sample in range(X.shape[0]):
			for ts in range(1, X.shape[1]):
				for track in range(X.shape[3]):
					if X[sample, ts, -1, track] == 1: # if held note is on
						X[sample, ts, :, track] = X[sample, ts-1, :, track]

		X = X[:, :, :-2, :]
		return X

	def get_guitar_bass_drums(self, song):
		guitar_tracks = []
		bass_tracks   = []
		drums_tracks  = []
		string_tracks = []

		for i, track in enumerate(song.tracks):
			if track.is_drum:
				track.name="Drums"
				drums_tracks.append(i)
			elif track.program >= 0 and track.program <= 31:
				track.name="Guitar"
				guitar_tracks.append(i)
			elif track.program >= 32 and track.program <= 39:
				track.name="Bass"
				bass_tracks.append(i)
			else:
				string_tracks.append(i)

		return guitar_tracks, bass_tracks, drums_tracks, string_tracks

	# preprocessing as in Hierarchical AE paper. (for lmd matched)
	def preprocess_dataset3(self, pianorolls_folder, metadata_folder, early_exit):
		# helper functions
		def msd_id_to_dirs(msd_id):
			"""Given an MSD ID, generate the path prefix.
			E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
			return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

		def msd_id_to_h5(h5):
			"""Given an MSD ID, return the path to the corresponding h5"""
			return os.path.join(metadata_folder,
								msd_id_to_dirs(msd_id) + '.h5')

		def check_four_fourth(time_sign):
			return time_sign.numerator == 4 and time_sign.denominator == 4

		# create necessary folders
		pianorolls_path = os.path.join(self.dataset_path, "pianorolls/")
		metadata_path = os.path.join(self.dataset_path, "metadata/")
		songs_path = os.path.join(self.dataset_path, "songs/")
		#instruments_path = os.path.join(self.dataset_path, "instruments/")
		dest_paths = [pianorolls_path, metadata_path, songs_path]#, instruments_path]
		for path in dest_paths:
			if not os.path.exists(path):
				os.makedirs(path)

		# count number of files of dataset (slow but ok)
		self.dataset_length = sum([len(files) for _, _, files in os.walk(pianorolls_folder)])
		# assign unique id for each song of dataset
		print("Preprocessing songs...")
		bar = progressbar.ProgressBar(max_value=self.dataset_length)
		pbc = 0
		yeah = 0
		fetch_meta = {} # in this dict I will store the id of the corresponding metadata file
		max_bar_silence = 0 

		for path, subdirs, files in os.walk(pianorolls_folder):
			for file in files:
				store_meta = False
				pbc += 1
				msd_id = path.split("/")[-1]
				filename = file.split(".")[0]

				
				if early_exit != None and pbc > early_exit:
					return

				# test 0: check keysignature = 4/4 always.
				try:
					pm_song = pm.PrettyMIDI(os.path.join(path, file))
				except Exception:
					continue

				if not all([check_four_fourth(tmp) for tmp in pm_song.time_signature_changes ]):
					continue

				del pm_song # don't need pretty midi object anymore, now i need pianorolls
				
				try:
					base_song = pproll.parse(os.path.join(path, file), beat_resolution=4)
				except Exception:
					continue

				# trova uno strumento chitarra, uno basso e uno batteria
				guitar_tracks, bass_tracks, drums_tracks, string_tracks = self.get_guitar_bass_drums(base_song)

				try:
					assert(string_tracks)
				except AssertionError:
					continue

				#if string_tracks:
				base_song.merge_tracks(string_tracks, mode="max", program=48, name="Strings", remove_merged=True)
				
				# merging tracks change order of them, need to re-find the new index of Trio track
				guitar_tracks, bass_tracks, drums_tracks, string_tracks = self.get_guitar_bass_drums(base_song)
				
				# take all possible combination of guitar, bass and drums
				for guitar_track in guitar_tracks:
					for bass_track in bass_tracks:
						for drums_track in drums_tracks:
							# select only trio tracks (and strings)
							current_tracks = [drums_track, bass_track, guitar_track, -1]
							names = ["Drums", "Bass", "Guitar", "Strings"]

							# create temporary song with only that tracks
							song = pproll.Multitrack()
							song.remove_empty_tracks()

							for i, current_track in enumerate(current_tracks):
								song.append_track(
									pianoroll=base_song.tracks[current_track].pianoroll,
									program=base_song.tracks[current_track].program,
									is_drum=base_song.tracks[current_track].is_drum,
									name=names[i]
								)

							song.beat_resolution = base_song.beat_resolution
							song.tempo = base_song.tempo

							song.binarize()
							song.assign_constant(1)

							# Test 1: check whether a track is silent during all the song
							if song.get_empty_tracks():
								continue
							
							pianoroll = song.get_stacked_pianoroll()

							i = 0
							while i + self.phrase_size <= pianoroll.shape[0]:
								window = pianoroll[i:i+self.phrase_size, :, :]
								# print("window from", i, "to", i+self.phrase_size)

								# keep only the phrases that have at most one bar of consecutive silence
								# for each track
								bar_of_silences = np.array([0] * self.n_tracks)
								for track in range(self.n_tracks):
									j = 0
									while j + self.bar_size <= window.shape[0]:
										if window[j:j+self.bar_size, :, track].sum() == 0:
											bar_of_silences[track] += 1

										j += 1#self.bar_size
								
								# if the phrase is good, let's store it
								if not any(bar_of_silences > max_bar_silence):
									# data augmentation, random transpose bar
									for shift in np.random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6], 1, replace=False):
										tmp = pproll.Multitrack()
										tmp.remove_empty_tracks()
										for track in range(self.n_tracks):
											tmp.append_track(
												pianoroll=window[:, :, track],
												program=song.tracks[track].program,
												name=config.instrument_names[song.tracks[track].program],
												is_drum=song.tracks[track].is_drum
											)

										tmp.beat_resolution = 4
										tmp.tempo = song.tempo
										tmp.name = str(yeah)

										tmp.transpose(shift)
										tmp.check_validity()
										tmp.save(os.path.join(pianorolls_path, str(yeah) + ".npz"))
										del tmp
										store_meta = True
										# adding link to corresponding metadata file
										fetch_meta[str(yeah)] = pbc
										yeah += 1

								i += self.bar_size
							del song

				# finished with pianorolls, storing rest (if needed)
				if store_meta:
					base_song.pad_to_multiple(self.phrase_size)
					base_song.pad_to_same()
					base_song.save(os.path.join(songs_path, str(pbc) + ".npz"))
					
					# fetching corresponding metadata from Million song dataset
					with tables.open_file(msd_id_to_h5(msd_id)) as h5:
						title = str(h5.root.metadata.songs.cols.title[0], "utf-8")
						artist = str(h5.root.metadata.songs.cols.artist_name[0], "utf-8")
						album = str(h5.root.metadata.songs.cols.release[0], "utf-8")
						genres = [ str(genre, "utf-8") for genre in list(h5.root.metadata.artist_terms[:]) ]

					metadata = {
						"title": title,
						"artist": artist,
						"album": album,
						"genres": genres
					}
					with open(os.path.join(metadata_path, str(pbc) + ".json"), "w") as fp:
						json.dump(metadata,  fp)

					# saving link to metadata dict
					with open(os.path.join(self.dataset_path, "meta_link.json"), "w") as fp:
						json.dump(fetch_meta, fp)

				del base_song
				bar.update(pbc)
			
		print("pbc:", pbc)
		print("yeah:", yeah)


	def extract_real_song_names(self, pianorolls_folder, metadata_folder, early_exit):
		# helper functions
		def msd_id_to_dirs(msd_id):
			"""Given an MSD ID, generate the path prefix.
			E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
			return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

		def msd_id_to_h5(h5):
			"""Given an MSD ID, return the path to the corresponding h5"""
			return os.path.join(metadata_folder,
								msd_id_to_dirs(msd_id) + '.h5')

		def check_four_fourth(time_sign):
			return time_sign.numerator == 4 and time_sign.denominator == 4

		# create necessary folders
		pianorolls_path = os.path.join(self.dataset_path, "pianorolls/")
		metadata_path = os.path.join(self.dataset_path, "metadata/")
		songs_path = os.path.join(self.dataset_path, "songs/")
		#instruments_path = os.path.join(self.dataset_path, "instruments/")
		dest_paths = [pianorolls_path, metadata_path, songs_path]#, instruments_path]
		for path in dest_paths:
			if not os.path.exists(path):
				os.makedirs(path)

		# count number of files of dataset (slow but ok)
		self.dataset_length = sum([len(files) for _, _, files in os.walk(pianorolls_folder)])
		# assign unique id for each song of dataset
		print("Extracting real song names...")
		bar = progressbar.ProgressBar(max_value=self.dataset_length)
		pbc = 0
		yeah = 0
		fetch_meta = {} # in this dict I will store the id of the corresponding metadata file
		max_bar_silence = 0 

		with open(os.path.join(self.dataset_path, "song_names.txt"), "a") as song_names_fp: 
			for path, subdirs, files in os.walk(pianorolls_folder):
				for file in files:
					store_meta = False
					pbc += 1
					msd_id = path.split("/")[-1]
					filename = file.split(".")[0]

					# early exit
					if early_exit != None and pbc > early_exit:
						return

					# test 0: check keysignature = 4/4 always.
					try:
						pm_song = pm.PrettyMIDI(os.path.join(path, file))
					except Exception:
						continue

					if not all([check_four_fourth(tmp) for tmp in pm_song.time_signature_changes ]):
						continue

					del pm_song # don't need pretty midi object anymore, now i need pianorolls
					
					try:
						base_song = pproll.parse(os.path.join(path, file), beat_resolution=4)
					except Exception:
						continue

					# trova uno strumento chitarra, uno basso e uno drums
					guitar_tracks, bass_tracks, drums_tracks, string_tracks = self.get_guitar_bass_drums(base_song)

					try:
						assert(string_tracks)
					except AssertionError:
						continue

					#if string_tracks:
					base_song.merge_tracks(string_tracks, mode="max", program=48, name="Strings", remove_merged=True)
					
					# merging tracks change order of them, need to re-find the new index of Trio track
					guitar_tracks, bass_tracks, drums_tracks, string_tracks = self.get_guitar_bass_drums(base_song)

					# take all possible combination of guitar, bass and drums
					for guitar_track in guitar_tracks:
						for bass_track in bass_tracks:
							for drums_track in drums_tracks:
								# select only trio tracks (and strings)
								current_tracks = [drums_track, bass_track, guitar_track, -1]
								names = ["Drums", "Bass", "Guitar", "Strings"]

								# create temporary song with only that tracks
								song = pproll.Multitrack()
								song.remove_empty_tracks()

								for i, current_track in enumerate(current_tracks):
									song.append_track(
										pianoroll=base_song.tracks[current_track].pianoroll,
										program=base_song.tracks[current_track].program,
										is_drum=base_song.tracks[current_track].is_drum,
										name=names[i]
									)

								song.beat_resolution = base_song.beat_resolution
								song.tempo = base_song.tempo

								# Test 1: check whether a track is silent during all the song
								if song.get_empty_tracks():
									continue

								pianoroll = song.get_stacked_pianoroll()

								i = 0
								while i + self.phrase_size <= pianoroll.shape[0]:
									window = pianoroll[i:i+self.phrase_size, :, :]

									# keep only the phrases that have at most one bar of consecutive silence
									# for each track
									bar_of_silences = np.array([0] * self.n_tracks)
									for track in range(self.n_tracks):
										j = 0
										while j + self.bar_size <= window.shape[0]:
											if window[j:j+self.bar_size, :, track].sum() == 0:
												bar_of_silences[track] += 1

											j += 1#self.bar_size
									
									# if the phrase is good, let's store it
									#print(bar_of_silences)
									if not any(bar_of_silences > max_bar_silence):
											store_meta = True
											# adding link to corresponding metadata file
											# yeah: pianorolls counter
											# pbc: song/metadata counter
											fetch_meta[str(yeah)] = pbc
											yeah += 1

									i += self.bar_size
								del song

					# finished with pianorolls, storing rest (if needed)
					if store_meta:
						# fetching corresponding metadata from Million song dataset
						with tables.open_file(msd_id_to_h5(msd_id)) as h5:
							title = str(h5.root.metadata.songs.cols.title[0], "utf-8")
							artist = str(h5.root.metadata.songs.cols.artist_name[0], "utf-8")
							album = str(h5.root.metadata.songs.cols.release[0], "utf-8")
							genres = [ str(genre, "utf-8") for genre in list(h5.root.metadata.artist_terms[:]) ]

						metadata = {
							"title": title,
							"artist": artist,
							"album": album,
							"genres": genres
						}

						song_names_fp.write(str(pbc) + " -> " + artist + " - " + title + "\n")
					del base_song

					bar.update(pbc)


	def count_genres(self, dataset_path, max_genres):
		max_pbc = sum([len(files) for _, _, files in os.walk(os.path.join(dataset_path, "songs"))])
		# assign unique id for each song of dataset
		print("Extracting real song names...")
		bar = progressbar.ProgressBar(max_value=max_pbc)
		pbc = 0
		counter = Counter()
		for path, subdirs, files in os.walk(os.path.join(dataset_path, "songs")):
			for song in files:
				pbc += 1

				song_number = song.split(".")[0]

				with open(os.path.join(dataset_path, "metadata", song_number + ".json")) as metadata_fp:
					metadata = json.load(metadata_fp)

				counter.update(metadata["genres"])				
				bar.update(pbc)

		print("Genres found:")
		pp.pprint(counter.most_common(max_genres))
		
		with open(os.path.join(dataset_path, "genre_counter.json"), "w") as fp:
			json.dump(counter.most_common(max_genres), fp)

		genres_list = [ x[0] for x in list(counter.most_common(max_genres)) ]
		
		if not os.path.exists(os.path.join(dataset_path, "labels")):
			os.makedirs(os.path.join(dataset_path, "labels"))

		# now generate labels information (S latents)
		print("Generating labels information...")
		bar = progressbar.ProgressBar(max_value=max_pbc)
		pbc = 0
		for path, subdirs, files in os.walk(os.path.join(dataset_path, "songs")):
			for song in files:
				pbc += 1

				song_number = song.split(".")[0]

				with open(os.path.join(dataset_path, "metadata", song_number + ".json")) as metadata_fp:
					metadata = json.load(metadata_fp)

				# setting corresponding tags
				label = np.zeros(max_genres)
				for genre in metadata["genres"]:
					try:
						idx = genres_list.index(genre)
						label[idx] = 1
					except ValueError:
						pass

				np.save(os.path.join(dataset_path, "labels", song_number), label)
				bar.update(pbc)