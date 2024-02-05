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

with $`B\in\mathbb{R}^{d\times r}, A\in\mathbb{R}^{r\times k}`$ and $`\alpha\in\mathbb{N}`$.

We seek to optimize the choice of 4 hyperparameters within this context: $`r`$, $`\alpha`$, the dropout probability of the optimizer AdamW and the learning rate of the fine-tuning. Each combination is denoted as a vector of hyperparameters $`\theta=(r,\alpha,dropout,lr)^ \top`$.

## Requirements

The HuggingFace API is central to our experiment. It implements language models, training and test procedures. See [Transformers](https://huggingface.co/docs/transformers/index) and [PEFT](https://huggingface.co/docs/peft/index) especially.


* Python >= 3.9
* All dependecies are listed in the `requirements.txt` file. Run
```bash
python3.9 -m pip install -r requirements.txt
```
to install all libraries at once.
* If you wish to use a LLaMA model through the HuggingFace Transformers API, you will need to be authorized by Meta AI. Follow the instructions on the HuggingFace webpage dedicated to the model you want to use (for the 7B version we used, see [here](https://huggingface.co/meta-llama/Llama-2-7b-hf)).
* Plan that you will need a significant amount of GPU memory. At GERAD, only the `atlas` server was able to run our experiments. You will need the A100 GPUs with 80Gb RAM (40Gb will not be enough).
    :white_check_mark: you CAN run an optimization with less than 4 GPUs but expect the computation to be slower.
    :x: you CANNOT run an optimization with any of the 40Gb RAM GPUs. The size of the model and the data will cause an overflow.
* NOMAD 4

## Repo organization

* `bbo` contains all files needed to reproduce the experiment described in [experiment 1](#exp1)

with NOMAD.
* `bbo

## The pipeline

## Already run experiments

Each experiment sets a specific objective function and uses specific sets of values for each variable.

<a id="exp1">### Experiment 1</a>
This experiment tries to solve
```math
    \max\limits_{\theta\in\Theta}\quad\text{MMLU}(\text{LoRA}(\text{LLaMA-2-7B},\mathcal{D},\theta))\enspace.
```

<a id="exp2">### Experiment 2</a>
This experiment tries to solve
```math
    \min_{\theta\in\Theta}\quad\mathcal{L}\left(\text{LoRA}(\text{LLaMA-2-7B},\theta),\mathcal{D}_{\text{valid}}\right)\enspace.
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

## References
<a id="Vas17">[Vas+17]</a> A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, L. Kaiser, & I. Polosukhin (2017). Attention is All You Need. In *Proceedings of the 31st International Conference on Neural Information Processing Systems* (pp. 6000–6010). Curran Associates Inc..

<a id="Hu21">[Hu+21]</a> E.J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, & W. Chen. (2021). LoRA: Low-Rank Adaptation of Large Language Models.

<a id="Touv23">[Touv+23]</a> H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. Canton Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. Singh Koura, M-A. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E.M. Smith, R. Subramanian, X.E. Tan, B. Tang, R. Taylor, A. Williams, J.X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, & T. Scialom. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models..