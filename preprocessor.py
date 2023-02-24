#!/usr/bin/python3

import argparse
from glob import glob
from os import path
from scipy import signal
from scipy.io.wavfile import read as _wavread
import numpy as np
from pyfilterbank import melbank
import pickle as pk

# melbank consts
A = melbank.compute_melmat(80,90,7600,1001)
As = (A[0].T / A[0].sum(axis=1)).T
A = (As, A[0]) # normalized, unnormalized

class sampledAudio:
    def __init__(self, file, pre_emph=True):
        """
        Parameters:
        filename
        sampling rate in Hz, default 16kHz
        """
        self.data = ""
        self.file = file
        self._readSample(pre_emph)

    def _readSample(self, pre_emph=True):
        aud = _wavread(self.file)

        if pre_emph:
            preem = aud[1].astype(np.float64)
            preem[1:] -= 0.97 * preem[:-1]
            self.data = preem.astype(np.int16)
        else:
            self.data = aud[1].astype(np.int16)

    def getSTFT(self, nfft=2000, noverlap=750, nperseg=1000,fs=1):
        spectro = signal.stft(self.data,fs=fs,nfft=nfft,noverlap=noverlap,nperseg=nperseg)
        spectro=spectro[2].T
        spectro = np.array(spectro)
        spectro = spectro.reshape(-1,nperseg+1)

        return spectro


class SpectrogramPreprocessor:

    def __init__(self, args, **av):
        speakers = av['speakers']
        self.write_emb_dict(args.embs, speakers)
        self.write_spectrograms(
            self.get_spectros_from_files(
                args.rootdir, speakers=speakers,
                parameters=(args.param_u,
                            args.param_v,
                            args.param_s,
                            args.param_t) ))

    def write_emb_dict(self, embs, tokens):
        if embs is None:
            return
        parsed = embs
        parsed = parsed.split(',')
        d, k = int(parsed[0]), int(parsed[1])
        emb_dict = {}

        i_token = 0

        for token in tokens:
            assert i_token <= d- k
            I = np.zeros(d)
            I[i_token:i_token + k] = 1.
            emb_dict[token] = I
            i_token += k

        with open(parsed[-1], 'wb') as fi:
            pk.dump(emb_dict, fi)

    def get_spectros_from_files(self, rootdir, **av):
        speakers = av['speakers']
        u,v,s,t = av['parameters']
        u,v,s,t = \
            float(u), float(v), float(s), float(t)

        files = []
        spectros = []

        for s in speakers:
            files += [ (s, glob(rootdir + "/" + s + "/*.wav")) ]

        minlevel = 1e-5

        for Fs in files:
            speaker_id, F = Fs
            specs_ = []
            for f in F:
                Y = sampledAudio(f, pre_emph=False).getSTFT().T

                Y = np.abs(Y.T) @ A[1].T

                Y = np.log10(np.maximum(Y, minlevel))

                D_db = (20 * Y) -16
                Q = (np.maximum(D_db - u * 100, minlevel) + v)/v
                Y = np.clip(Q, 0, t)
                Y = (Y - 1) / (t - 1)
                specs_.append(Y)

            spectros += [(speaker_id, specs_)]

        return spectros

    def write_spectrograms(self, obj):
        with open(args.outfile, 'wb') as fi:
            pk.dump(obj, fi)

if __name__ == "__main__":
    print("hello")
    parser = argparse.ArgumentParser(
        description='''
Perform STFT and compression of wav files into spectrograms suitable for autoencoding.  Kindly ensure data are structured into folders /rootdir/speaker_id/files.wav and that no extraneous files or directories are in /rootdir

Plus a baseline log-floor of 1e-5, the following transformation is applied to the log spectrogram X:

f(X) = ( |X - 100u| + v ) / v + s
clipped to [0, t]
    ''', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('rootdir', metavar='rootdir', type=str,
                        help='directory containing wav files')

    parser.add_argument('-o', dest='outfile', type=str,
                        default='/tmp/spectrograms.pk',
                        help='path to transformed spectrograms file (default /tmp/spectrograms.pk')

    parser.add_argument('-e', dest='embs',
                        metavar='d,k,embs.pk',
                        type=str,
                        default=None,
                        help=\
"""
Generate d-dimensional one hot embeddings for n speakers with 2k dimensions spanning each speaker.  n is inferred from contents of rootdir
these are saved in emb.pk.

ex. n = 2, d = 8, k = 2

Speaker 1 has embedding [0,0,1,1,0,0,0,0]
Speaker 2 has embedding [0,0,0,0,0,0,1,1]

""")

    parser.add_argument('-u', dest='param_u', default=0.,
                        help="additive noise floor (default 0.)")
    parser.add_argument('-v', dest='param_v', default=100.,
                        help="compression ratio (default 100.)")
    parser.add_argument('-s', dest='param_s', default=0.,
                        help="subtractive noise floor (default 0.)")
    parser.add_argument('-t', dest='param_t', default=1.,
                        help="clipping constant (default 1.)")
    parser.add_argument('-I', dest='tokens', default=None,
                        help=\
"""
IDs of speakers to include, separated by comma -I id_1,id_2,id_3
Default behavior is to omit -I flag, and include every speaker in /rootdir.  This compiles spectrograms in all subdirectories ie. /rootdir/speaker_id/***.wav

""")

    args = parser.parse_args()

    path.join(args.rootdir, "*")

    if args.tokens is None:
        speakers = [
            path.split(p)[-1]  for p in glob(path.join(args.rootdir, "*")) ]
    else:
        speakers = \
            args.tokens.split(',')
    SpectrogramPreprocessor(args, speakers=speakers)
