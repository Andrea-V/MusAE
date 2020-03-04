# MusÆ: Adversarial Autoencoder for learning style-aware Music representations.

This repository provides the Python implementation of MusAE, an [Adversarial Autoencoder](https://arxiv.org/abs/1511.05644) able to generate and modify musical MIDI sequences in a meaningful way. This model is written in Keras, using TensorFlow backend. You can learn more about MusAE by reading [this paper on ArXiv](https://arxiv.org/abs/2001.05494). You can also listen to some of the results that I've uploaded on [my YouTube channel](https://www.youtube.com/playlist?list=PLxrPCQsIK9XVVpTIun9meuPcOdWaG-aSg).

## Getting Started

- File *main.py* is the main script. It basically just calls the appropriate functions of other scripts, according to the chosen configuration settings.
- File *config.py* contains the model's configuration settings.
- Files *train.py*, *train_gm.py*, *encoders.py*,  *decoders.py*, *discriminator.py*,  contains the actual models' implementations and training processes.
- File *dataset.py* contains helper functions used for preprocessing, postprocessing and accessing the dataset.

## Installation/Dependencies

The main dependencies are:
- Keras 2.2.2
- TensorFlow 1.12

plus a number of other various python libraries. You should be able to install everything using the *musae.yml* file provided in the repository.

## Usage

First, you need to create the dataset. You need to download a set of MIDI files and then create the corresponding pianorolls. The scripts in *dataset.py* are already tuned to preprocess the [MIDI Lakh Dataset](https://colinraffel.com/projects/lmd/), so I suggest you to use that. But you can use any set of MIDI files you want. In order to start the preprocessing phase, you should set the configuration variable *preprocessing* to *True* in *config.py*. Then run *main.py*. The preprocessing phase may take some time, even days, depending on your dataset's size.

After the dataset has been created, you can actually train the model. Just set *preprocessing=False* and re-run *main.py*. Training can, again, take a long time, depending on the actual dataset size and on the model architecture. Be sure to instatiate the appropriate version of MusAE (single gaussian or gaussian mixture) with the specific training file.

In *config.py* you can freely change many model parameters, from the number of latent variable to the number of hidden layers of the many architectural components. The parameters name should be quite auto-explicatives.


## Getting Help

For any other additional information, you can email me at andrea.valenti@phd.unipi.it.

## Contributing Guidelines  

You're free (and encouraged) to make your own contributions to this project.

## Code of Conduct

Just be nice, and respecful of each other!

## License

All source code from this project is released under the MIT license.
