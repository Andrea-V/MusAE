import config
from midi_aae import MidiAAE
from midi_dataset_2 import MidiDataset
import os
import numpy as np
import pretty_midi as pm
import pypianoroll as pproll
from keras import backend as K
import keras
import pprint
pp = pprint.PrettyPrinter(indent=4)


if __name__ == "__main__":
	dataset = MidiDataset(**config.midi_params, **config.general_params)

	if config.preprocessing:
		print("Preprocessing dataset...")
		#dataset.preprocess_dataset3("lmd_matched", "lmd_matched_h5", early_exit=5000)
		dataset.count_genres(config.general_params["dataset_path"], max_genres=config.model_params["s_length"])
		dataset.create_batches(batch_size=config.training_params["batch_size"])
		#dataset.extract_real_song_names("lmd_matched", "lmd_matched_h5", early_exit=3000)
		exit(-1)

	print("Initialising MIDI-AAE...")
	aae = MidiAAE(**config.model_params, **config.midi_params, **config.general_params, **config.training_params)

	print("Training MIDI-AAE...")
	aae.train_v2(dataset)
