#!/usr/bin/python3

import os
import pickle as pk

import tensorflow as tf

from .data_loader import Sample_Loader, shuffled_Sample_Loader
from .autovc_model import AutoVC

def main():
    model = AutoVC(inputs = XGENERATOR)
    model.compile(optimizer=tf.keras.optimizers.Adam(.0001))

    model.fit(shuf_XGENERATOR,   epochs=300000, shuffle=False)

if __name__ == "__main__":
    # load preprocessed spectrograms
    PWD = os.getcwd()
    with open( os.path.join(PWD, "bin", "embs.pk"), 'rb') as fi:
        emb_dict = pk.load(fi)

    with open( os.path.join(PWD, "bin", "spects.pk"), 'rb') as fi:
        spectros = pk.load(fi)

    with open( os.path.join(PWD, "bin", "valspects.pk"), 'rb') as fi:
        valspectros = pk.load(fi)

    speakers = list(emb_dict.keys())

    # generate training sequence
    XGENERATOR = Sample_Loader(spectros, emb_dict, batch_size=2)
    shuf_XGENERATOR = shuffled_Sample_Loader(XGENERATOR)

    main()
