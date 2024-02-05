# Hyperparameter Optimization for the Instruction-Tuning of Large Language Models with NOMAD 4

[Paper](https://doi.org/10.48550/arXiv.2312.00949)

This repo gathers the work that has been done by the BBO team @ GERAD for the Hyperparameter Optimization (HPO) project in the context of the Alliance with *Huawei-Canada* between september 2023 and february 2024.

## Starting point

The PDF document `approach.pdf` in the `Docs` folder thoroughly describes the theory behind our work, as well as our approach. It is recommended to read it first in order to fully understand what is undertaken here.

## Approach

```math
W_0x+\Delta Wx\simeq W_0x+\frac{\alpha}{r}BAx\enspace.
```

## Contact

For any questions about the theory or the code presented in this repo, you may contact:
* sacha.benarroch-lelong@zaclys.net
* christophe.tribes@polymtl.ca

# Description of the experiments

## BBO3

### Parameters
* Rank: choice in [4,8,16,32]
* Alpha: integer proportional to the rank in [rank, 64 * rank]
* Dropout: choice in [0, 0.0001, 0.001, 0.01, 0.1]
* Learning rate: fixed to 10e-3.5

### Initial point

* Rank: 8
* Alpha: 8
* Dropout: 0