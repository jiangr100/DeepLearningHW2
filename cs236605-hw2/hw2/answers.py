r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    lr = 0.02
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    lr_vanilla = 0.02
    lr_momentum = 0.005
    lr_rmsprop = 0.0002
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    
    lr = 0.005
    
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""

1. it's not what i expected, which is that higher dropout rate would result in less of overfitting, 
but here in all of the 3 graphs, we all have a pretty high overfitting. This I think is due to a
small batch size, so no matter how it will be easy to overfit the data even if we already dropped
lots of features.

    Also, the accuracy is getting lower along with the increasing of the dropout rate, which I think
might due to a small amount of features, so we accidentally dropped lots of essential features we 
actually need.

2. as i mentioned above, it's still more easy to overfit using a low dropout rate, even it's still
considered high. And the accuracy is lower due to the droping of essential features.


"""

part2_q2 = r"""

the test loss of CE is not completely related to the test accuracy, since there might be the case 
that we have a really high evaluation value of a certain prediction (for example 6 and 0), so we 
will result in a big loss here, but actually the model works well on not extreme datas (like when 
people write numbers clearly), so the accuracy is pretty high as well.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
