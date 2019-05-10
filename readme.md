# MusAE: Adversarial Autoencoder for Symbolic Music Generation

This repository provides the Python implementation of MusAE, an [Adversarial Autoencoder](https://arxiv.org/abs/1511.05644) able to generate and modify musical MIDI sequences in a meaningful way. This model is written in Keras, using TensorFlow backend. You can learn more about MusAE by consulting my Master's thesis on [my LinkedIn profile](https://www.linkedin.com/in/avalenti93/). You can also listen to some of the results that I've uploaded on [my YouTube channel](https://www.youtube.com/playlist?list=PLxrPCQsIK9XVVpTIun9meuPcOdWaG-aSg).

**UPDATE:** I am currently developing a Pytorch implementation of MusAE, it will be avaible soon on my GitHub page.

## Getting Started

- File *main.py* is the main script. It basically just calls the appropriate functions of other scripts, according to the chosen configuration settings.
- File *config.py* contains the model's configuration settings.
- File *midi_aae.py* contains the actual model implementation.
- File *midi_dataset_2.py* contains helper functions used for preprocessing, postprocessing and accessing the dataset.

## Installation/Dependencies

The main dependencies are:
- Keras 2.2.2
- TensorFlow 1.12

plus a number of other python libraries. You should be able to install everything via pip (using a separate conda environment is highly recommended). 

## Usage

First, you need to create the dataset. You need to download a set of MIDI files and then create the corresponding pianorolls. The scripts in *midi_dataset_2.py* are already tuned to preprocess the [MIDI Lakh Dataset](https://colinraffel.com/projects/lmd/), so I suggest you to use that. But you can use any set of MIDI files you want. In order to start the preprocessing phase, you should set the configuration variable *preprocessing* to *True* in *config.py*. Then run *main.py*. The preprocessing phase may take some time, even days, depending on your dataset's size.

After the dataset has been created, you can actually train the model. Just set *preprocessing=False* and re-run *main.py*. Training can, again, take a long time, depending on the actual dataset size and on the model architecture.

In *config.py* you can freely change many model parameters, from the number of latent variable to the number of hidden layers of the many architectural components. The parameters name should be quite auto-explicatives.

In order to produce some actual results during training, you should uncomment some lines in *midi_aae.py* (from line 618 downwards, more or less). You can also select your own song or musical sequences by changing the *idx* argument in the saving function calls.

## Getting Help

For any other additional information, you can email me at valentiandrea@rocketmail.com.

## Contributing Guidelines  

You're free (and encouraged) to make your own contributions to this project.

## Code of Conduct

Just be nice, and respecful of each other!

## License

All source code from this project is released under the MIT license.
