# Choco-SGD
This repository provides code for **communication-efficient decentralized ML training** (both deep learning, compatible with [PyTorch](https://pytorch.org/), and traditional convex machine learning models.

We provide code for the main experiments in the papers 
 - [Decentralized Stochastic Optimization and Gossip Algorithms with Compressed Communication](https://arxiv.org/abs/1902.00340) and 
 - [Decentralized Deep Learning with Arbitrary Communication Compression](https://arxiv.org/abs/1907.09356).

Please refer to the folders `convex_code` and `dl_code` for more details.


# References
If you use the code, please cite the following papers:

```
@inproceedings{koloskova2019choco,
    title = {Decentralized Stochastic Optimization and Gossip Algorithms with Compressed Communication},
    author = {Anastasia Koloskova and Sebastian U. Stich and Martin Jaggi},
    booktitle = {ICML 2019 - Proceedings of the 36th International Conference on Machine Learning},
    url = {http://proceedings.mlr.press/v97/koloskova19a.html},
    publisher = {PMLR}, 
    volume = {97},
    pages = {3479--3487},
    year = {2019}
}
```
and 
```
@inproceedings{koloskova2020decentralized,
  title={Decentralized Deep Learning with Arbitrary Communication Compression},
  author={Anastasia Koloskova* and Tao Lin* and Sebastian U Stich and Martin Jaggi},
  booktitle={ICLR 2020 - International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=SkgGCkrKvH}
}
```
