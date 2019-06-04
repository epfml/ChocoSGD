# CHOCO-SGD

Here we provide the code for reproducing main experiments from the paper [Decentralized Stochastic Optimization and Gossip Algorithms with Compressed Communication](https://arxiv.org/abs/1902.00340)

### Load dataset

First you need to download datasets from LIBSVM library and convert them into pickle format. For that from
```
cd data
wget -t inf https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
wget -t inf https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2
cd ../code
python pickle_datasets.py
```
If you get memory error, you can leave rcv1 dataset in the sparse format, but this would slow down training time.


For running experiments with `epsilon` dataset
```
python experiment_epsilon_final.py final
```
