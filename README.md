# keras-autovc
#### Keras implementation of AutoVC

This repository provides an implementation of the AutoVC voice conversion autoencoder using Keras.  Check out the original PyTorch implementation at https://github.com/auspicious3000/autovc based on the paper https://arxiv.org/abs/1905.05879.


#### Current [x] and planned [ ] features
  - [x] Data pipeline:
	  - [x] Wav-file reading
	  - [x] -> Spectrogram cleaning 
	  - [x] -> Training set generation
  - [x] Speaker embedder
	  - [x] (one-hot) Many-to-many conversion (seen speakers)
	  - [ ] (GE2E) Zero-shot conversion (unseen speakers)
	  
#### Audio demo
Some examples of converted audio can be found at https://www.alexwei.net/keras-autovc-demo
