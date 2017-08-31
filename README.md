# Noisy-A3C-Keras

Initial A3C implementation from https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/

Noisy implementation inspiration from https://github.com/Kaixhin/NoisyNet-A3C

Paper https://arxiv.org/abs/1706.10295

The noise needs to be resampled after every training epoch, and I would suggest setting the noise to 0 when the training is done. Examples are in the code.
