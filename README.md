# Scoop: An Optimization Algorithm for Profiling Attacks against Higher-Order Masking
Authors: Anonymized

This repository contains the implementation of the Scoop algorithm for profiling attacks against higher-order masking, as well as supplementary materials for additional figures and reproducing the results of the paper.

## Requirements

To avoid any conflicts, we recommend using a virtual environment. To create a new virtual environment, run the following commands:

```conda create --name scoop_ches25 --file requirements.txt```

```conda activate scoop_ches25```

WARNING: Using `requirements.txt` assumes you have a CUDA-compatible GPU with CUDA>=12.1 installed. If you do not have a GPU, or if you have a different version of CUDA, you may need to install the appropriate version of PyTorch manually.

## Usage

At this time, \textsc{Scoop} has only been implemented in PyTorch. To use \textsc{Scoop}, one needs to import \texttt{scoop.py} in his project:

```python
    from scoop import Scoop
```

\texttt{Scoop} is a class that inherits from \texttt{torch.optim.Optimizer}. Hence, one can use it as any other optimizer in PyTorch, we detail its hyperparameters later. The main difference with standard optimizer is that \textsc{Scoop} relies on a Hessian estimator. Hence, \texttt{Scoop} has a \texttt{hutchinson\_hessian} method that update the Hessian estimation in-place. The entire training loop is then:

The main contribution is located in `src/scoop.py`. To use Scoop, create an instance of the `Scoop` class as you would do with any other optimizer:

```python
optimizer = Scoop(model.parameters(), lr=lr)
```

The training loop is then slightly modified to include the Hessian computation and the Scoop update:

```python
...
        loss = F.nll_loss(Y_pred, Y_batch)/math.log(2)
        loss.backward(create_graph=True)
        if iter % hessian_update == hessian_update - 1:
            optimizer.hutchinson_hessian() # SCOOP SPECIFIC LINE
        optimizer.step()
        train_loss += loss.item()
...
```

In case the update is too costly, one can decide to update the Hessian estimation every $k$ iterations (meaning $k$ mini-batches). This should not hinder the performance of the algorithm too much and is actually used in some second-order optimization algorithms. The different hyperparameters of **Scoop** are:

### Hyperparameters of **Scoop**

| **Hyperparameter**     | **Description**                      | **Default**   | **Suggested Range**                   |
|-------------------------|--------------------------------------|---------------|----------------------------------------|
| `lr`                   | Learning rate                       | 1e-4          | [1e-5, 1e-2]                          |
| `betas`                | Momentum parameters                 | (0.965, 0.99) | [0.9, 0.999]                          |
| `weight_decay`         | $\ell_2$ regularization             | 0             | [0, 0.3]                              |
| `estimator`            | Hessian estimator                   | "biased_hutchinson" | ["classic", "biased_hutchinson"] |
| `hessian_iter`         | # of iterations for Hessian estimator | 5            | As much as you can afford[^1]         |

[^1]: One iteration is already much better than **Adam**.

While default values are given, we suggest adding those hyperparameters to the fine-tuning search grid. Additional hyperparameters can be added to the optimizer, for example $\epsilon$ where $\psi(x) = \|x\|_{1+\epsilon}$. It is by default set to 0.1, but in case you face a problem where sparsity in $\mathbf{F}$ is not desired, you can set $\epsilon \geq 1$. $\epsilon = 1$ would be Newton's method approximation of **Scoop**, and would behave similarly to the Hutchinson variant of Liu *et al.* work~\cite{liu2023sophia}.

## Examples

You can explore the different notebooks in this repository for more detailed examples and additional figures.

## Citation

If you use Scoop, or this code, in your research please cite the following paper:

```bibtex
TBD --> anonymized
```