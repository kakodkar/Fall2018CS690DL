"""© 2018 Jianfei Gao All Rights Reserved"""
import os
import math
import numpy as np
import torch


class PTBDataset(object):
    def __init__(self, path, dictionary):
        r"""Initialize the class

        Args
        ----
        root : Str
            Path of a PTB dataset file.
        dictionary : Dict
            Dictionary from words to integers.

        """
        # allocate data buffer
        self.seq = []

        self.str2int = dictionary

        # read feature file
        seq = []
        file = open(path, 'r')
        content = file.readlines()
        file.close()
        for line in content:
            # parse each line
            if len(line) == 0:
                continue
            else:
                sentence = line.strip().split() + ['<eos>']
            for word in sentence:
                if word in self.str2int:
                    pass
                else:
                    self.str2int[word] = len(self.str2int)
                seq.append(self.str2int[word])

        # formalize data sequences
        self.seq = np.array(seq, dtype=int)

        # statistics
        print('=' * 29)
        print("PTB: {:<25s}".format(path))
        print('-' * 29)
        print("Dictionary: {:>17s}".format("@{}".format(id(dictionary))))
        print('# Word IDs: {:>17d}'.format(len(dictionary)))
        print("# Words   : {:>17d}".format(len(self.seq)))
        print('=' * 29)
        print()

class MarkovChainLoader(object):
    def __init__(self, dataset, order):
        r"""Initialize the class

        Args
        ----
        dataset : PTBDataset
            A raw PTB dataset.
        order : Int
            Markov chain order.

        """
        raise NotImplementedError # missing

class BPTTBatchLoader(object):
    def __init__(self, dataset, bptt=35, batch_size=20):
        r"""Initialize the class

        Args
        ----
        dataset : PTBDataset
            A raw PTB dataset.
        bptt : Int
            Length of truncated backpropagation through time.
        batch_size : Int
            Batch size.

        """
        # truncate dataset for batch splitting
        seq = dataset.seq
        num_words = len(seq)
        num_batches = num_words // batch_size
        num_words = num_batches * batch_size
        seq = seq[0:num_words]

        raise NotImplementedError # missing

    def __len__(self):
        r"""Length of the class"""
        return self.num_bptt

    def __iter__(self):
        r"""Iterator of the class"""
        return self.Iterator(self)

    class Iterator(object):
        def __init__(self, loader):
            r"""Initialize the class"""
            self.loader = loader
            self.ptr = 0

        def __len__(self):
            r"""Length of the class"""
            return len(self.loader)

        def __next__(self):
            r"""Next element of the class"""
            # validate next element
            if self.ptr >= self.loader.num_bptt:
                raise StopIteration
            else:
                pass

            # update pointers in raw data for next element
            ptr0 = self.ptr * self.loader.bptt
            ptr1 = min(ptr0 + self.loader.bptt, len(self.loader.seq) - 1)
            self.ptr += 1

            # get input and target batch
            input_batch = self.loader.seq[ptr0:ptr1]
            target_batch = self.loader.seq[ptr0 + 1:ptr1 + 1]
            input_batch = torch.LongTensor(input_batch).contiguous()
            target_batch = torch.LongTensor(target_batch).contiguous()
            return input_batch, target_batch

if __name__ == '__main__':
    dictionary = {}
    train_dataset = PTBDataset('./data/ptb/ptb.train.txt', dictionary)
    valid_dataset = PTBDataset('./data/ptb/ptb.valid.txt', dictionary)
    test_dataset = PTBDataset('./data/ptb/ptb.test.txt', dictionary)
    train_loader = MarkovChainLoader(train_dataset, 3)
    valid_loader = MarkovChainLoader(valid_dataset, 3)
    test_loader = MarkovChainLoader(test_dataset, 3)
    train_loader = BPTTBatchLoader(train_dataset)
    valid_loader = BPTTBatchLoader(valid_dataset)
    test_loader = BPTTBatchLoader(test_dataset, batch_size=1)