import numpy as np
import random
import tensorflow as tf

class Sample_Loader(tf.keras.utils.Sequence):
    def __init__(self, speaker_sets, emb_dict,  batch_size, convert=None, seed=None):
        # each speaker gets a tuple (speaker_id,   sample spectrograms) -> speaker_sets
        self.speaker_sets = speaker_sets
        self.emb_dict = emb_dict
        self.batch_size = batch_size
        self.len_crop = 128
        self.convert = convert

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        assert 1

    def __len__(self):
        return len(self.speaker_sets)

    def __getitem__(self, idx):
        s_id, list_uttrs = self.speaker_sets[idx]

        if not self.seed is None:
            self.rng = np.random.default_rng(self.seed)

        a = self.rng.integers(0, len(list_uttrs), size=self.batch_size)

        cropped_uttrs = []
        for i in a:
            tmp = list_uttrs[i]

            if tmp.shape[0] < self.len_crop:
                len_pad = self.len_crop - tmp.shape[0]
                uttr = np.pad(tmp, ((0,len_pad),(0,0)), 'constant')
            elif tmp.shape[0] > self.len_crop:
                left = np.random.randint(tmp.shape[0]-self.len_crop)
                uttr = tmp[left:left+self.len_crop, :]
            else:
                uttr = tmp
            cropped_uttrs.append(uttr)

        cropped_uttrs = np.array(cropped_uttrs)
        emb_vects = [
            self.emb_dict[s_id] for i in a]
        if self.convert is None:
            to_vects = emb_vects
        else:
            to_vects = [
                self.emb_dict[self.convert] for i in a]

        return { 'xim': cropped_uttrs,
                 'embeds': np.array(emb_vects),
                 'embeds_': np.array(to_vects) } ,\
                 cropped_uttrs


class shuffled_Sample_Loader(tf.keras.utils.Sequence):
    def __init__(self,loader):
        self.loader = loader

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        idx_ = random.randint(0,1)
        return self.loader.__getitem__(idx_)
