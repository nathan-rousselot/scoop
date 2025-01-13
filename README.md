# Scoop: An Optimization Algorithm for Profiling Attacks against Higher-Order Masking
Authors: Anonymized

This repository contains the implementation of the Scoop algorithm for profiling attacks against higher-order masking, as well as supplementary materials for additional figures and reproducing the results of the paper.

## Requirements

To avoid any conflicts, we recommend using a virtual environment. To create a new virtual environment, run the following commands:

```conda create --name scoop_ches25 --file requirements.txt```
```conda activate scoop_ches25```

## Usage

The main contribution is located in `src/scoop.py'. To use Scoop, create an instance of the `Scoop` class as you would do with any other optimizer:

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

You can explore the different notebooks in this repository for more detailed examples and additional figures.

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
TBD --> anonymized
```