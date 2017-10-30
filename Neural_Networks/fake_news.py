import argparse
import chainer
from chainer import cuda, utils, Variable
import chainer.functions as F
import chainer.links as L
import cPickle as pickle
import json
import numpy as np
import pandas as pd
import random
import re
import string
import sys
import unicodedata
from collections import Counter
import math

'''
Explanation of code:
The main idea behind training the neural network was to
tweak the parameters as to fully utilize all of the data and
train it through a sizable number of epochs. During the
different iterations I used several combinations and evaluated
both through perplexity and qualitatively. The code below 
has an additional layer than what was originally given
as well as and added mechanism to estimate perplexity.

The first experiment was to utilize 10 epoch, 128 hidden units, 6 millionwords in training and a 
threshold of 50. This led to a perplexity of 20 and 
qualitatively didn't offer really good results. 

The next experiment was to increase the number of hidden
units to 256, everything else left the same. This result 
decreased perplexity to 16.

The next experiment was to increase the number of hidden 
units to 512 and the size of the testing data from 60,000 to
120,000, this result decreased perplexity to 10

The next experiment was to increase the number of  
hidden layers to 1024, this failed due to a memory error

The next experiment was to return the number of hidden layers to 512, and increase the number of epochs to 100
in order to allow the data to converge better. 

This was the model that best performed in terms of perpexity, it was 5, as well as 
qualitative results. 
This was the full list of parameters:

python fake_news.py -u char -g 0 -b 3500 -tl 6000000 -vl 120000 -e 100 -gl 1500 -nu 512 -o char6m_th400_nu512_epoch100.txt -th 400
'''

class FooRNN(chainer.Chain):
    """ Two Layer LSTM """
    def __init__(self, n_vocab, n_units, train=True):
        super(FooRNN, self).__init__(  # input must be a link
            embed=L.EmbedID(n_vocab, n_units),
            l1 = L.LSTM(n_units, n_units),
            l2 = L.LSTM(n_units, n_units),
            
            #Added an extra layer
            l3 = L.LSTM(n_units, n_units),
            
            l4 = L.Linear(n_units, n_vocab),
        )
        self.n_vocab = n_vocab
        self.n_units = n_units
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        h3 = self.l3(F.dropout(h2, train=self.train))
        y= self.l4(F.dropout(h3, train=self.train))
        return y

def read_data(category='b', unit='char', thresh=50):
    fname = '/project/cmsc25025/uci-news-aggregator/{cat}_article.json'.format(
        cat=category
    )
    raw_doc = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            text = json.loads(line)['text']
            if len(text.split()) >= 100:
                raw_doc.append(
                    unicodedata.normalize('NFKD', text)
                    .encode('ascii', 'ignore').lower()
                    .translate(string.maketrans("\n", " "))
                    .strip()
                )

    raw_doc = ' '.join(raw_doc)

    if unit == 'char':
        vocab = {el: i for i, el in enumerate(set(raw_doc))}
        id_to_word = {i: el for el, i in vocab.iteritems()}
    else: # unit == 'word':
        raw_doc = re.split('(\W+)', raw_doc)
        count = Counter(raw_doc)

        vocab = {}
        ii = 0
        for el in count:
            if count[el] >= thresh:
                vocab[el] = ii
                ii += 1

        id_to_word = {i: el for el, i in vocab.iteritems()}


    doc = [vocab[el] for el in raw_doc if el in vocab]
    print '  * doc length: {}'.format(len(doc))
    print '  * vocabulary size: {}'.format(len(vocab))
    sys.stdout.flush()

    return doc, vocab, id_to_word


def convert(data, batch_size, ii, gpu_id=-1):
    xp = np if gpu_id < 0 else cuda.cupy
    offsets = [t * len(data) // batch_size for t in xrange(batch_size)]
    x = [data[(offset + ii) % len(data)] for offset in offsets]
    x_in = chainer.Variable(xp.array(x, dtype=xp.int32))
    y = [data[(offset + ii + 1) % len(data)] for offset in offsets]
    y_in = chainer.Variable(xp.array(y, dtype=xp.int32))
    return x_in, y_in


def gen_text(model, curr, id_to_word, text_len, gpu_id=-1):
    xp = np if gpu_id < 0 else cuda.cupy

    n_vocab = len(id_to_word)
    gen = [id_to_word[curr]] * text_len
    model.predictor.reset_state()
    for ii in xrange(text_len):
        output = model.predictor(
            chainer.Variable(xp.array([curr], dtype=xp.int32))
        )
        p = F.softmax(output).data[0]
        if gpu_id >= 0:
            p = cuda.to_cpu(p)
        curr = np.random.choice(n_vocab, p=p)
        gen[ii] = id_to_word[curr]

    return ''.join(gen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=2048,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu_id', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train_len', '-tl', type=int, default=5000000,
                        help='training doc length')
    parser.add_argument('--valid_len', '-vl', type=int, default=50000,
                        help='validation doc length')
    parser.add_argument('--gen_len', '-gl', type=int, default=1000,
                        help='generated doc length')
    parser.add_argument('--bp_len', '-bl', type=int, default=30,
                        help='back propagate length')
    parser.add_argument('--unit', '-u', type=str, default='char',
                        help='type of unit in doc')
    parser.add_argument('--n_units', '-nu', type=int, default=512,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--n_text', '-nt', type=int, default = 10,
                        help='Number of generated news')
    parser.add_argument('--output', '-o', type=str, default='output.txt',
                        help='file to write generated txt')
    parser.add_argument('--thresh', '-th', type=int, default=50,
                        help='threshold of words counts for vocabulary')
    args = parser.parse_args()

    gpu_id = args.gpu_id
    n_epoch = args.epoch
    train_len = args.train_len
    valid_len = args.valid_len
    batch_size = min(args.batch_size, args.train_len)

    print "loading doc...."
    sys.stdout.flush()

    doc, vocab, id_to_word = read_data(
      category='b', unit=args.unit, thresh=args.thresh
    )
    n_vocab = len(vocab)

    if train_len + valid_len > len(doc):
        raise Exception(
            'train len {} + valid len {} > doc len {}'.format(
                train_len, valid_len, len(doc)
            )
        )
    train = doc[:train_len]
    valid = doc[(train_len+1):(train_len+1+valid_len)]

    print "initializing...."
    sys.stdout.flush()
    model = L.Classifier(FooRNN(n_vocab, args.n_units, train=True))
    sys.stdout.flush()
    model.predictor.reset_state()
    #optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(100))

    if gpu_id >= 0:
        cuda.get_device(gpu_id).use()
        model.to_gpu()

    # main training loop
    print "training loop...."
    sys.stdout.flush()
    xp = np if gpu_id < 0 else cuda.cupy
    for t in xrange(n_epoch):
        train_loss = train_acc = n_batches = loss = 0
        model.predictor.reset_state()
        for i in range(0, len(train) // batch_size + 1):
            x, y = convert(train, batch_size, i, gpu_id)
            batch_loss = model(x, y)
            loss += batch_loss
            if (i+1) % min(len(train) // batch_size, args.bp_len) == 0:
                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                optimizer.update()
            train_loss += batch_loss.data
            n_batches += 1
        train_loss = train_loss / n_batches
        train_acc = train_acc / n_batches

        # validation
        valid_loss = valid_acc = n_batches = 0
        for i in range(0, len(valid) // batch_size + 1):
            x, y = convert(valid, batch_size, i, gpu_id)
            batch_loss = model(x, y)
            valid_loss += batch_loss.data
            n_batches += 1
        valid_loss = valid_loss / n_batches
        valid_acc = valid_acc / n_batches
        

        print '  * Epoch {} train loss={} valid loss={}'.format(
            t,
            train_loss,
            valid_loss
        )
        sys.stdout.flush()

        if t >= 1 and xp.abs(train_loss - old_tr_loss) / train_loss < 1e-5:
            print "Converged."
            sys.stdout.flush()
            break

        old_tr_loss = train_loss
        
        ###
    # Original code, calculating perplexity
    # In this section we are calculating perplexity as
    # the exponent of the loss function. We are doing this
    # because we know the loss is the cross entropy between y predicted
    # and the true distribution, and we know that:
    
    # perplexity = exp( cross_entropy)
    # so entropy is exp(loss)
    
    entropy = math.exp(valid_loss) 
    print ' Perplexity is {}'.format(entropy)

    print "generating doc...."
    sys.stdout.flush()
    model.predictor.train = False
    with open(args.output, 'w') as f:
        for ii in xrange(args.n_text):
            start = random.choice(xrange(len(vocab)))
            fake_news = gen_text(
                model,
                start,
                id_to_word,
                text_len=args.gen_len,
                gpu_id=gpu_id
            )
            f.write(fake_news)
            f.write('\n\n\n')

    if gpu_id >= 0:
        model.to_cpu()
    with open('model_%s.pickle' % args.output[:-4], 'wb') as f:
        pickle.dump(model, f, protocol=2)


if __name__ == '__main__':
    main()
