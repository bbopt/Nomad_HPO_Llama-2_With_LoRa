# Hyperparameter Optimization for the Instruction-Tuning of Large Language Models with NOMAD 4

[Paper](https://doi.org/10.48550/arXiv.2312.00949)

This repo gathers the work that has been done by the BBO team @ GERAD for the Hyperparameter Optimization (HPO) project in the context of the Alliance with *Huawei-Canada* between september 2023 and february 2024.

## Starting point

The PDF document `approach.pdf` in the `Docs` folder thoroughly describes the theory behind our work, as well as our approach. It is recommended to read it first in order to fully understand what is undertaken here.

In the following, the reader is assumed to be familiar with the theory developed in this document, especially with: the Transformers architecture [[Vas+17]](#Vas17), the concept of instruction-tuning, the LoRA fine-tuning method [[Hu+21]](#Hu21), 4 widespread NLP tests (MMLU, BBH, DROP and HumanEval) and the family of LLaMA language models. [[Touv+23]](#Touv23)

## Approach

We perform the fine-tuning of the 7B parameter variant of LLaMA 2 on a 53k-sized dataset of instructions with the LoRA fine-tuning method. Recall that the LoRA method relies on the following quasi-equality:

```math
W_0x+\Delta Wx\simeq W_0x+\frac{\alpha}{r}BAx\enspace.
```

with `$B\in\mathbb{R}^{d\times r}, A\in\mathbb{R}^{r\times k}$` and `$\alpha\in\mathbb{N}$`.

We seek to optimize the choice of 4 hyperparameters within this context: `$r$`, `$\alpha$`, the dropout probability of the optimizer AdamW and the learning rate of the fine-tuning. Each combination is denoted as a vector of hyperparameters `\theta=(r,\alpha,dropout,lr)^ \top`.

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

## References
<a id="Vas17">[Vas+17]</a> A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, L. Kaiser, & I. Polosukhin (2017). Attention is All You Need. In *Proceedings of the 31st International Conference on Neural Information Processing Systems* (pp. 6000–6010). Curran Associates Inc..

<a id="Hu21">[Hu+21]</a> E.J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, & W. Chen. (2021). LoRA: Low-Rank Adaptation of Large Language Models.

<a id="Touv23">[Touv+23]</a> H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. Canton Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. Singh Koura, M-A. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E.M. Smith, R. Subramanian, X.E. Tan, B. Tang, R. Taylor, A. Williams, J.X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, & T. Scialom. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models..