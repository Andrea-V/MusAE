import config
import matplotlib.pyplot as plt
from train import MusAE
from train_gm import MusAE_GM
from dataset import MidiDataset
import os
import numpy as np
import pretty_midi as pm
import pypianoroll as pproll
from keras import backend as K
import keras
import pprint
from mpl_toolkits.mplot3d import Axes3D

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
	dataset = MidiDataset(**config.midi_params, **config.general_params, **config.preprocessing_params)

	if config.preprocessing:
		print("Preprocessing dataset...")
		# set early_exit to None to preprocess entire dataset
		dataset.preprocess_dataset3("lmd_matched", "lmd_matched_h5", early_exit=None)
		dataset.count_genres(config.preprocessing_params["dataset_path"], max_genres=config.model_params["s_length"])
		dataset.create_batches(batch_size=config.training_params["batch_size"])
		dataset.extract_real_song_names("lmd_matched", "lmd_matched_h5", early_exit=None)
		exit(-1)

	print("Initialising MusÆ...")
	aae = MusAE_GM(**config.model_params, **config.midi_params, **config.general_params, **config.training_params)
	
	print("Training MusÆ...")
	aae.train_v2(dataset)