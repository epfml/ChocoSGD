import argparse
import multiprocessing as mp
import os
import pickle
from sklearn.datasets import load_svmlight_file

import numpy as np

from logistic import LogisticDecentralizedSGD
from parameters import Parameters
from utils import pickle_it

from experiment import run_logistic, run_experiment

A, b = None, None

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('experiment', type=str)
  args = parser.parse_args()
  print(args)
  assert args.experiment in ['final']

  dataset_path = os.path.expanduser('../data/epsilon.pickle')
  n, d = 400000, 2000

###############################################
### RANDOM DATA PARTITION #####################
###############################################
  n_cores = 9
################### FINAL ################################

  split_way = 'random'
  split_name = split_way

  num_epoch = 10
  n_repeat = 5
  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [
                Parameters(name="decentralized-exact", num_epoch=num_epoch,
                           lr_type='decay', initial_lr=0.1, tau=d,
                           regularizer=1 / n, quantization='full',
                           n_cores=n_cores, method='plain',
                           split_data_random_seed=random_seed,
                           distribute_data=True, split_data_strategy=split_name,
                           topology='ring', estimate='final'),
      ]
    run_experiment("dump/epsilon-final-decentralized-" + split_way + "-" +\
                   str(n_cores) + "/", dataset_path, params, nproc=10)

  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [
                Parameters(name="centralized", num_epoch=num_epoch,
                           lr_type='decay', initial_lr=0.1, tau=d,
                           regularizer=1 / n, quantization='full',
                           n_cores=n_cores, method='plain',
                           split_data_random_seed=random_seed,
                           distribute_data=True, split_data_strategy=split_name,
                           topology='centralized', estimate='final')]
    run_experiment("dump/epsilon-final-centralized-" + split_way + "-" + str(n_cores)\
                   + "/", dataset_path, params, nproc=10)


  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [
                Parameters(name="decentralized-top-20", num_epoch=num_epoch,
                           lr_type='decay', initial_lr=0.1, tau=d,
                           regularizer=1 / n, consensus_lr=0.04,
                           quantization='top', coordinates_to_keep=20,
                           n_cores=n_cores, method='choco', topology='ring',
                           estimate='final', split_data_random_seed=random_seed,
                           distribute_data=True, split_data_strategy=split_name)]
    run_experiment("dump/epsilon-final-choco-top-20-" + split_way + "-" + str(n_cores)\
                   + "/", dataset_path, params, nproc=10)


  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [
                Parameters(name="decentralized-random-20", num_epoch=num_epoch,
                           lr_type='decay', initial_lr=0.1, tau=d,
                           regularizer=1 / n, consensus_lr=0.01,
                           quantization='random-biased', coordinates_to_keep=20,
                           n_cores=n_cores, method='choco', topology='ring',
                           estimate='final', split_data_random_seed=random_seed,
                           distribute_data=True, split_data_strategy=split_name)]
    run_experiment("dump/epsilon-final-choco-random-20-" + split_way+ "-" + str(n_cores)\
                   + "/", dataset_path, params, nproc=10)


  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [
                Parameters(name="decentralized-qsgd-8", num_epoch=num_epoch,
                           lr_type='decay', initial_lr=0.1, tau=d,
                           regularizer=1 / n, consensus_lr=0.34,
                           quantization='qsgd-biased', num_levels=16,
                           n_cores=n_cores, method='choco', topology='ring',
                           estimate='final', split_data_random_seed=random_seed,
                           distribute_data=True, split_data_strategy=split_name)]
    run_experiment("dump/epsilon-final-choco-qsgd-4bit-" + split_way + "-" + str(n_cores)\
                   + "/", dataset_path, params, nproc=10)

  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [Parameters(name="dcd-psgd-random-20",
                           num_epoch=num_epoch, lr_type='decay',
                           initial_lr=1e-15, tau=d, regularizer=1 / n,
                           quantization='random-unbiased', coordinates_to_keep=20,
                           n_cores=n_cores, method='dcd-psgd',
                           split_data_random_seed=random_seed,
                           distribute_data=True, split_data_strategy=split_name,
                           topology='ring', estimate='final')]
    run_experiment("dump/epsilon-final-dcd-random-20-" + split_way + "-" + str(n_cores)\
                   + "/", dataset_path, params, nproc=10)

  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [Parameters(name="ecd-psgd-random", num_epoch=num_epoch,
                            lr_type='decay', initial_lr=1e-6, tau=d,
                            regularizer=1 / n, consensus_lr=None,
                            quantization='random-unbiased',
                            coordinates_to_keep=20, n_cores=n_cores,
                            method='ecd-psgd', split_data_random_seed=random_seed,
                            distribute_data=True, split_data_strategy=split_name,
                            topology='ring', estimate='final')]
    run_experiment("dump/epsilon-final-ecd-random-20-" + split_way + "-" + str(n_cores)\
                   + "/", dataset_path, params, nproc=10)

###################### qsgd quantization #####################################

  
  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [Parameters(name="dcd-psgd-qsgd", num_epoch=num_epoch,
                            lr_type='decay', initial_lr=0.01, tau=d,
                            regularizer=1 / n, quantization='qsgd-unbiased',
                            num_levels=16, n_cores=n_cores, method='dcd-psgd',
                            split_data_random_seed=random_seed,
                            distribute_data=True, split_data_strategy=split_name,
                            topology='ring', estimate='final')]
    run_experiment("dump/epsilon-final-dcd-qsgd-4bit-" + split_way + "-" + str(n_cores)\
                   + "/", dataset_path, params, nproc=10)

  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [Parameters(name="ecd-psgd-qsgd", num_epoch=num_epoch,
                            lr_type='decay', initial_lr=1e-06, tau=d,
                            regularizer=1 / n, quantization='qsgd-unbiased',
                            num_levels=16, n_cores=n_cores, method='ecd-psgd',
                            split_data_random_seed=random_seed,
                            distribute_data=True, split_data_strategy=split_name,
                            topology='ring', estimate='final')]
    run_experiment("dump/epsilon-final-ecd-qsgd-4bit-" + split_way + "-" + str(n_cores)\
                   + "/", dataset_path, params, nproc=10)




###############################################
### SORTED DATA PARTITION #####################
###############################################
################################ FINAL ####################################




  split_way = 'sorted'
  split_name = split_way
  if split_way == 'sorted':
    split_name = 'label-sorted'

  n_repeat = 5
  num_epoch = 10
  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [
                Parameters(name="decentralized-exact", num_epoch=num_epoch,
                           lr_type='decay', initial_lr=0.1, tau=d,
                           regularizer=1 / n, quantization='full',
                           n_cores=n_cores, method='plain',
                           split_data_random_seed=random_seed,
                           distribute_data=True, split_data_strategy=split_name,
                           topology='ring', estimate='final'),
      ]
    run_experiment("dump/epsilon-final-decentralized-" + split_way+ "-" + str(n_cores)\
                   + "/", dataset_path, params, nproc=10)

  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [
                Parameters(name="centralized", num_epoch=num_epoch, lr_type='decay',
                           initial_lr=0.1, tau=d, regularizer=1 / n,
                           quantization='full', n_cores=n_cores, method='plain',
                           split_data_random_seed=random_seed, distribute_data=True,
                           split_data_strategy=split_name, topology='centralized',
                           estimate='final')]
    run_experiment("dump/epsilon-final-centralized-" + split_way+ "-" + str(n_cores)\
                   + "/", dataset_path, params, nproc=10)


  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [
                Parameters(name="decentralized-top-20", num_epoch=num_epoch, lr_type='decay',
                           initial_lr=0.1, tau=d, regularizer=1 / n, consensus_lr=0.04,
                           quantization='top', coordinates_to_keep=20, n_cores=n_cores,
                           method='choco', topology='ring', estimate='final',
                           split_data_random_seed=random_seed, distribute_data=True,
                           split_data_strategy=split_name, random_seed=40 + random_seed)]
    run_experiment("dump/epsilon-final-choco-top-20-" + split_way+ "-" + str(n_cores) + "/",
        dataset_path, params, nproc=10)


  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [
                Parameters(name="decentralized-random-20", num_epoch=num_epoch, lr_type='decay',
                           initial_lr=0.1, tau=d, regularizer=1 / n, consensus_lr=0.01,
                           quantization='random-biased', coordinates_to_keep=20, n_cores=n_cores,
                           method='choco', topology='ring', estimate='final',
                           split_data_random_seed=random_seed, distribute_data=True,
                           split_data_strategy=split_name, random_seed=60 + random_seed)]
    run_experiment("dump/epsilon-final-choco-random-20-" + split_way+ "-" + str(n_cores) + "/",
          dataset_path, params, nproc=10)


  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [
                Parameters(name="decentralized-qsgd-8", num_epoch=num_epoch, lr_type='decay',
                           initial_lr=0.1, tau=d, regularizer=1 / n, consensus_lr=0.34,
                           quantization='qsgd-biased', num_levels=16, n_cores=n_cores,
                           method='choco', topology='ring', estimate='final',
                           split_data_random_seed=random_seed, distribute_data=True,
                           split_data_strategy=split_name)]
    run_experiment("dump/epsilon-final-choco-qsgd-4bit-" + split_way + "-" + str(n_cores) + "/",
                   dataset_path, params, nproc=10)


  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [Parameters(name="dcd-psgd-random-20", num_epoch=num_epoch,
                            lr_type='decay', initial_lr=1e-15, tau=d,
                            regularizer=1 / n, quantization='random-unbiased',
                            coordinates_to_keep=20, n_cores=n_cores, method='dcd-psgd',
                            split_data_random_seed=random_seed, distribute_data=True,
                            split_data_strategy=split_name, topology='ring',
                            estimate='final')]
    run_experiment("dump/epsilon-final-dcd-random-20-" + split_way + "-" + str(n_cores) + "/",
                   dataset_path, params, nproc=10)

  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [Parameters(name="ecd-psgd-random",
                           num_epoch=num_epoch, lr_type='decay',
                           initial_lr=1e-10, tau=d, regularizer=1 / n,
                           quantization='random-unbiased', coordinates_to_keep=20,
                           n_cores=n_cores,
                           method='ecd-psgd', split_data_random_seed=random_seed,
                           distribute_data=True,
                           split_data_strategy=split_name,
                           topology='ring', estimate='final')]
    run_experiment("dump/epsilon-final-ecd-random-20-" + split_way + "-" + str(n_cores) + "/",
                   dataset_path, params, nproc=10)

###################### qsgd quantization #####################################

  
  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [Parameters(name="dcd-psgd-qsgd",
                           num_epoch=num_epoch, lr_type='decay',
                           initial_lr=0.01, tau=d, regularizer=1 / n,
                           quantization='qsgd-unbiased', num_levels=16, n_cores=n_cores,
                           method='dcd-psgd', split_data_random_seed=random_seed,
                           distribute_data=True, split_data_strategy=split_name,
                           topology='ring', estimate='final')]
    run_experiment("dump/epsilon-final-dcd-qsgd-4bit-" + split_way + "-" + str(n_cores) + "/",
                   dataset_path, params, nproc=10)

  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [Parameters(name="ecd-psgd-qsgd",
                           num_epoch=num_epoch, lr_type='decay',
                           initial_lr=1e-12, tau=d, regularizer=1 / n,
                           quantization='qsgd-unbiased', num_levels=16, n_cores=n_cores,
                           method='ecd-psgd', split_data_random_seed=random_seed,
                           distribute_data=True, split_data_strategy=split_name,
                           topology='ring', estimate='final')]
    run_experiment("dump/epsilon-final-ecd-qsgd-4bit-" + split_way + "-" + str(n_cores) + "/",
                   dataset_path, params, nproc=10)


