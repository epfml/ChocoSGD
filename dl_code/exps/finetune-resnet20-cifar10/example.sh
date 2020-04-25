# centralized sgd with complete topology.
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet20 --optimizer sgd \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory True \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda True --use_ipc False --comm_device cuda \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 150,225 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --hostfile iccluster/hostfile --graph_topology complete --track_time True --display_tracked_time True \
    --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --mpi_path $HOME/.openmpi/ --evaluate_avg True

# decentralized sgd with ring topology.
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet20 --optimizer sgd \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory True \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda True --use_ipc False --comm_device cuda \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 150,225 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --hostfile iccluster/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --mpi_path $HOME/.openmpi/ --evaluate_avg True

# parallel_choco with sign + norm for ring topology
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet20 --optimizer parallel_choco \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory True \
    --batch_size 128 --base_batch_size 24 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda True --use_ipc False --comm_device cuda \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 150,225 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op sign --consensus_stepsize 0.45 --compress_ratio 0.9 --quantize_level 16 --is_biased True \
    --hostfile iccluster/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --mpi_path $HOME/.openmpi/ --evaluate_avg True

# parallel_choco with sign + norm for social topology
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet20 --optimizer parallel_choco \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory True \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 32 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda True --use_ipc False --comm_device cuda \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 150,225 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op sign --consensus_stepsize 0.5 --compress_ratio 0.9 --quantize_level 16 --is_biased True \
    --hostfile iccluster/hostfile --graph_topology social --track_time True --display_tracked_time True \
    --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --mpi_path $HOME/.openmpi/ --evaluate_avg True

# dcd_psgd with quantize_qsgd for ring topology
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet20 --optimizer dcd_psgd \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory True \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda True --use_ipc False --comm_device cuda \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 150,225 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op quantize_qsgd --compress_ratio 0.9 --quantize_level 16 --is_biased True \
    --hostfile iccluster/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --mpi_path $HOME/.openmpi/ --evaluate_avg True

# ecd_psgd with compress_top_k for ring topology
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet20 --optimizer ecd_psgd \
    --avg_model True --experiment test \
    --data cifar10 --pin_memory True \
    --batch_size 128 --base_batch_size 64 --num_workers 0 --eval_freq 1 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 8 --n_sub_process 1 --world 0,0,0,0,0,0,0,0 --on_cuda True --use_ipc False --comm_device cuda \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_schedule_scheme custom_multistep --lr_change_epochs 150,225 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op compress_top_k --compress_ratio 0.9 --quantize_level 16 --is_biased True \
    --hostfile iccluster/hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --mpi_path $HOME/.openmpi/ --evaluate_avg True
