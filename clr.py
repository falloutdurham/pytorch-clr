import numpy as np
import warnings
from torch.optim.optimizer import Optimizer


class CyclicLearningRate(object):
    """This class (partially) implements the 'triangular' and 'triangular2'
    polices found in Leslie N. Smith's [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
    paper. It alters the learning rate on a per-epoch basis between a minimum learning
    rate and a maximum learning rate, depending on a cycle defined by the initialization
    parameters.

    LIMITATIONS: Ideally, the learning rate should be changed on a per-batch basis
    rather than an epoch. I may produce a new version based using [torchsample](https://github.com/ncullen93/torchsample)
    to get easy access to per-batch callbacks, but I also wanted to produce a pure PyTorch version. 
    
    Args:
        verbose: 0 - quiet, 1 - print updates to learning rates
        min_lr: Lower bound on learning rate
        max_lr: Upper bound on learning rate
        iterations: How many iterations in your epoch
        stepsize: The stepsize of the triangular cycle (2-8 * iterations is a good guide)
        policy: 'triangular' or 'triangular2' from the CLR paper. Default is 'triangular'
        
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = scheduler = CyclicLearningRate(optimizer, min_lr=0.001, max_lr=0.015, iterations=60, stepsize=180)
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_acc, val_loss = validate(...)
        >>>     scheduler.step(val_loss, epoch)

    Note: code borrows idea from [Jiaming Liu's scheduler](https://github.com/Jiaming-Liu/pytorch-lr-scheduler)

    """

    policy_fns = { 'triangular': lambda x: 1,
                  'triangular2': lambda x: 1/(2.**(x-1)) }

    def __init__(self, optimizer, min_lr, max_lr, iterations, stepsize,policy='triangular', 
                 verbose=0):
        super(CyclicLearningRate, self).__init__()

        self.stepsize = stepsize
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.verbose = verbose
        self.policy = policy
        self.policy_fn = self.policy_fns[policy]
        self.iterations = iterations
        self.optimizer = optimizer

    def step(self, metrics, epoch):
        current = metrics        
        
        for param_group in self.optimizer.param_groups:
            current_lr = float(param_group['lr'])

            cycle = np.floor(1 + (float(self.iterations * epoch) / (2 * self.stepsize)))
            x = np.abs((float(self.iterations * epoch) / self.stepsize - (2 * cycle) + 1))
            new_lr = self.min_lr + ((self.max_lr - self.min_lr) * np.maximum(0, (1-x)) * self.policy_fn(cycle))
            param_group['lr'] = new_lr
            if(self.verbose > 0):
                print('CyclicLearningRate: Epoch %05d: Cycle is: %s, x is %s, new_lr is %s (policy: %s)' % (epoch, cycle, x, new_lr, self.policy))
