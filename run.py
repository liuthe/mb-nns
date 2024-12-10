#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backflow Model
run.py: Run Script for Simple NNB Model

Usage:
    run.py train [options]
    run.py decode [options] MODEL_PATH OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --max-train-iter=<int>                  max train iter [default: 1000]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""
import math
import sys
import pickle
import time
from pathlib import Path


from docopt import docopt

from model.nnb_model import NNB, Hypothesis
from observable.hamiltonian import Hubbard
from sampler.sampler import generate_sample

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils.batch_utils import batch_iter

import torch
import torch.nn.utils
from torch.utils.tensorboard import SummaryWriter


def evaluate_ppl(nnb_model, H_model, dev_data, batch_size=50):
    """ Evaluate energy on dev sentences
    @param nnb_model (NNB): NNB Model
    @param H_model (Hubbard): Hubbard model
    @param dev_data (Tensor of shape (num_samples, 2*L*L)): Tensor containing the samples
    @param batch_size (batch size)
    @returns energy (energy of the given samples)
    """
    was_training = nnb_model.training
    nnb_model.eval()

    energy = 0. 

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = H_model.energy_local(src_sents, tgt_sents)

            energy += loss.item()

    if was_training:
        nnb_model.train()

    return energy / batch_size


def train(args: Dict):
    """ Train the NNB Model.
    @param args (Dict): args from cmd line
    """
    train_batch_size = int(args['--batch-size'])
    max_train_iter = int(args['--max-train-iter'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    n_chains = 16
    n_samples = train_batch_size // n_chains

    t = 1
    U = 0
    L = 4
    sys_size = L * L
    hidden_size = 256
    num_fillings = [7, 7]
    model = NNB(sys_size = sys_size, 
                hidden_size = hidden_size, 
                num_fillings = num_fillings)
    h_model = Hubbard(L, t=t, U=U)
    
    tensorboard_path = "nnb_cuda" if args['--cuda'] else "nnb_local"
    writer = SummaryWriter(log_dir=f"./runs/{tensorboard_path}")
    model.train()

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)                       # EDIT: SMALLER LEARNING RATE

    num_trial = 0
    train_iter = patience = report_loss = 0
    report_examples = epoch = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin NNB self training')

    while True:
        epoch += 1

        #train_data, prob_list = generate_sample(model, n_samples, n_chains)
        #for batch_data in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
        for train_iter in range(max_train_iter):
            #train_iter += 1
            optimizer.zero_grad()
            batch_data, prob_list = generate_sample(model, n_samples, n_chains)
            batch_size = batch_data.size(0)

            batch_loss = h_model.local_energy(batch_data, model)
            batch_loss.backward()

            # clip gradient
            #grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss = batch_losses_val

            report_examples = batch_size

            if train_iter % log_every == 0:
                writer.add_scalar("loss/train", report_loss, train_iter)
                print('epoch %d, iter %d, avg. loss %.2f, ' \
                        'speed %.2f configs/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                        report_loss,
                                                                        report_examples / (time.time() - train_time),
                                                                        time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                #report_loss = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                writer.add_scalar("loss/val", batch_losses_val, train_iter)
                print('epoch %d, iter %d, cum. loss %.2f' % (epoch, train_iter,
                                                                batch_losses_val), file=sys.stderr)

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                valid_metric = batch_losses_val

                print('validation: iter %d, dev. energy %f' % (train_iter, batch_losses_val), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def decode(args: Dict[str, str]):
    """ Compute the ground state energy of the given neural network model.
    @param args (Dict): args from cmd line
    """
    seed = int(time.time())  # get the time right now to be the random seed
    print(f"Setting random seed: {seed}", file=sys.stderr)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = NNB.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    h_model = Hubbard(4, t=1, U=1)
    sample, prob_list = model.generate_sample(16,256)

    Path(args['OUTPUT_FILE']).parent.mkdir(parents=True, exist_ok=True)

    with open(args['OUTPUT_FILE'], 'w') as f:
        energy = model.local_energy(sample, h_model)
        f.write(str(energy.item()) + '\n')

def main():
    """ Main func.
    """
    start_time = time.time()

    args = docopt(__doc__)

    # Check pytorch version
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    main()
