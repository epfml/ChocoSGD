# CHOCO-SGD
Deep Learning code for the main experiments of the paper [Decentralized Deep Learning with Arbitrary Communication Compression](https://arxiv.org/abs/1907.09356).


## Getting started
Our experiments heavily rely on `Docker` and `Kubernetes`. For the detailed experimental environment setup, please refer to dockerfile under the `environments` folder.


### Use case of distributed training (centralized/decentralized)
Some simple explanation of the arguments used in the code.
* Arguments related to *distributed training*:
    * The `n_mpi_process` and `n_sub_process` indicates the number of nodes and the number of GPUs for each node. The data-parallel wrapper is adapted and applied locally for each node.
        * Note that the exact mini-batch size for each MPI process is specified by `batch_size`, while the mini-batch size used for each GPU is `batch_size/n_sub_process`.
    * The `world` describes the GPU topology of the distributed training, in terms of all GPUs used for the distributed training.
    * The `hostfile` from `mpi` specifies the physical location of the MPI processes.
    * We provide two use cases here:
        * `n_mpi_process=2`, `n_sub_process=1` and `world=0,0` indicates that two MPI processes are running on 2 GPUs with the same GPU id. It could be either 1 GPU at the same node or two GPUs at different nodes, where the exact configuration is determined by `hostfile`.
        * `n_mpi_process=2`, `n_sub_process=2` and `world=0,1,0,1` indicates that two MPI processes are running on 4 GPUs and each MPI process uses GPU id 0 and id 1 (on 2 nodes).
* Arguments related to *communication compression*:
    * The `graph_topology` 
    * The `optimizer` will decide the type of distributed training, e.g., centralized SGD, decentralized SGD
    * The `comm_op` specifies the communication compressor we can use, e.g., `sign+norm`, `random-k`, `top-k`.
    * The `choco_consenus_stepsize` determines the `consenus_stepsize` for `parallel_choco`.
* Arguments related to *learning*:
    * The `lr_schedule_scheme` and `lr_change_epochs` indicates that it is a stepwise learning rate schedule, with decay factor `10` for epoch `150` and `225`.
    * The `lr_scaleup`, `lr_warmup` and `lr_warmup_epochs` will decide if we want to scale up the learning rate, or warm up the learning rate.

The script below trains `ResNet-20` with `CIFAR-10`, as an example of decentralized training algorithm `parallel_choco` with `sign+norm` communication compressor.
```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet20 --optimizer parallel_choco \
    --avg_model True --experiment demo \
    --data cifar10 --pin_memory True \
    --batch_size 128 --base_batch_size ${base_batch_size[j]} --num_workers 2 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 16 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda True --use_ipc False --comm_device cuda \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 150,225 --lr_decay 10 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op sign --choco_consenus_stepsize 0.5 --compress_ratio 0.9 --quantize_level 16 --is_biased True \
    --hostfile iccluster/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --mpi_path $HOME/.openmpi/ --evaluate_avg True
```