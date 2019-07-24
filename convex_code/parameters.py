class Parameters:
    """
    Parameters class used for all the experiments, redefine a string representation to summarize the experiment
    """

    def __init__(self,
                 num_epoch,
                 lr_type,
                 initial_lr=None,
                 regularizer=None,
                 epoch_decay_lr=None,
                 consensus_lr=None,
                 quantization="full",
                 # number of coordinates k in top-k or random-k quantization
                 coordinates_to_keep=None,
                 # number of levels in qsgd quantization
                 num_levels=None,
                 estimate='final',
                 name=None,
                 # number of machines
                 n_cores=1,
                 topology='centralized',
                 method='choco',
                 distribute_data=False,
                 # whether each machine gets random data or continuous set of data
                 # might not have any difference, depends on the dataset
                 split_data_strategy=None,
                 tau=None,
                 real_update_every=1,
                 random_seed=None,
                 split_data_random_seed=None,
                 ):
        # a lot of sanity checks to fail fast if we have inconsistent parameters
        assert num_epoch >= 0
        assert lr_type in ['constant', 'epoch-decay', 'decay', 'bottou']

        if lr_type in ['constant', 'decay']:
            assert initial_lr > 0
        if lr_type == 'decay':
            assert initial_lr and tau
            assert regularizer > 0
        if lr_type == 'epoch-decay':
            assert epoch_decay_lr is not None
        if method in ['choco']:
            assert consensus_lr > 0
        else:
            assert consensus_lr is None

        assert quantization in ['full', 'top', 'random-biased', 'random-unbiased',
                                'qsgd-biased', 'qsgd-unbiased']
        if quantization == 'full':
            assert not coordinates_to_keep
        elif quantization in ['top', 'random-biased', 'random-unbiased']:
            assert coordinates_to_keep > 0
        else:
            assert num_levels > 0

        assert estimate in ['final', 'mean', 't+tau', '(t+tau)^2']

        assert n_cores > 0

        assert topology in ['centralized', 'ring', 'torus', 'disconnected']

        assert method in ['choco', 'dcd-psgd', 'ecd-psgd', 'plain']
        if method in ['dcd-psgd', 'ecd-psgd']:
            assert quantization in ['random-unbiased', 'qsgd-unbiased']

        if not distribute_data:
            assert not split_data_strategy
        else:
            assert split_data_strategy in ['naive', 'random', 'label-sorted']

        self.num_epoch = num_epoch
        self.lr_type = lr_type
        self.initial_lr = initial_lr
        self.regularizer = regularizer
        self.epoch_decay_lr = epoch_decay_lr
        self.consensus_lr = consensus_lr
        self.quantization = quantization
        self.coordinates_to_keep = coordinates_to_keep
        self.num_levels = num_levels
        self.estimate = estimate
        self.name = name
        self.n_cores = n_cores
        self.topology = topology
        self.tau = tau
        self.real_update_every = real_update_every
        self.random_seed = random_seed
        self.method = method
        self.distribute_data = distribute_data
        self.split_data_strategy = split_data_strategy
        self.split_data_random_seed = split_data_random_seed

    def __str__(self):
        if self.name:
            return self.name

        lr_str = self.lr_str()
        sparse_str = self.sparse_str()

        reg_str = ""
        if self.regularizer:
            reg_str = "-reg{}".format(self.regularizer)

        return "epoch{}-{}{}-{}-{}".format(self.num_epoch, lr_str, reg_str, sparse_str, self.estimate)

    def lr_str(self):
        lr_str = ""
        if self.lr_type == 'constant':
            lr_str = "lr{}".format(self.initial_lr)
        elif self.lr_type == 'decay':
            lr_str = "lr{}decay{}".format(self.initial_lr, self.epoch_decay_lr)
        elif self.lr_type == 'custom':
            lr_str = "lr{}/lambda*(t+{})".format(self.initial_lr, self.tau)
        elif self.lr_type == 'bottou':
            lr_str = "lr-bottou-{}".format(self.initial_lr)
        else:
            lr_str = "lr-{}".format(self.lr_type)

        return lr_str

    def sparse_str(self):
        sparse_str = self.quantization
        if quantization != 'full':
            sparse_str += "{}".format(self.coordinates_to_keep)
        return sparse_str

    def __repr__(self):
        return "Parameter('{}')".format(str(self))
