# CHOCO-SGD

Code for the main experiments of the paper [Decentralized Stochastic Optimization and Gossip Algorithms with Compressed Communication](https://arxiv.org/abs/1902.00340). 

### Datasets and Setup

First you need to download datasets from LIBSVM library and convert them into pickle format. For that from
```
cd data
wget -t inf https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
wget -t inf https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2
cd ../code
python pickle_datasets.py
```
If you get memory error, you can leave rcv1 dataset in the sparse format, but this would slow down training time.

### Reproduce the results

For running experiments with the `epsilon` dataset
```
python experiment_epsilon_final.py final
```


# Reference
If you use this code, please cite the following [paper](http://proceedings.mlr.press/v97/koloskova19a.html):

    @inproceedings{ksj2019choco,
      author = {Anastasia Koloskova and Sebastian U. Stich and Martin Jaggi},
      title = {Decentralized Stochastic Optimization and Gossip Algorithms with Compressed Communication},
      booktitle = {ICML 2019 - Proceedings of the 36th International Conference on Machine Learning},
      url = {http://proceedings.mlr.press/v97/koloskova19a.html},
      series = {Proceedings of Machine Learning Research},
      publisher = {PMLR}, 
      volume = {97},
      pages = {3479--3487},
      year = {2019}
    }
