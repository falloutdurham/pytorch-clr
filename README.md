# pytorch-clr
Port of Cyclic Learning Rates to PyTorch

This class (partially) implements the 'triangular' and 'triangular2' polices found in Leslie N. Smith's [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) paper. It alters the learning rate between a minimum learning rate and a maximum learning rate, depending on a cycle defined by the initialization parameters.

## Limitations

Ideally, the learning rate should be changed on a per-batch basis rather than an epoch, so you'll have to add in scaffolding into your train() methods to do that…or else just run this on a per-epoch basis for simplicity. I may produce a new version based using [torchsample](https://github.com/ncullen93/torchsample) to get easy access to per-batch callbacks, but I also wanted to produce a pure PyTorch version. 
    
## Usage    
    
Args:
* verbose: 0 - quiet, 1 - print updates to learning rates
* min_lr: Lower bound on learning rate
* max_lr: Upper bound on learning rate
* iterations: How many iterations in your epoch
* stepsize: The stepsize of the triangular cycle (2-8 * iterations is a good guide)
* policy: 'triangular' or 'triangular2' from the CLR paper. Default is 'triangular'
        
Example:

```
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> scheduler = scheduler = CyclicLearningRate(optimizer, min_lr=0.001, max_lr=0.015, iterations=60, stepsize=180)
>>> for epoch in range(10):
>>>     train(...)
>>>     val_acc, val_loss = validate(...)
>>>     scheduler.step(val_loss, epoch)
```
Note: code borrows idea from [Jiaming Liu's scheduler](https://github.com/Jiaming-Liu/pytorch-lr-scheduler)
