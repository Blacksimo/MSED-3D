# MSED-3D
Implementation of the state-of-the-art multi-channel 3D Convolutional sound event detection model in Keras. [Link to the reference paper](paper/MSED-3D.pdf)

## The Dataset
* Binaural audio, TUTSED 2017
* Street context audio at 24 bit and 44.1 kHz as sampling rate
* I manual annotation of 6 sound event classes: brakes squeaking, car, children, large vehicle, people speaking, and people walking

## The Implemented Features
* Log mel-band energy (MBE)
...Intra-Channel Features
* I Generalized cross correlation with phase transform (GCC)
...Inter-Channel Features

## Custom Metric
In order to match the required metric for the early stopping, a custom metric class has been implemented.

## Results and More
Results values, graphs and further information can be found in [Report Paper](report/pdf/NN.pdf) and in the related [Presentation Slides](report/pdf/Presentazione_NN.pdf)
