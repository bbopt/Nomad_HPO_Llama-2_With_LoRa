# Hyperparameter Optimization for the Instruction-Tuning of Large Language Models with NOMAD 4

[Paper](https://doi.org/10.48550/arXiv.2312.00949)

This repo gathers the work that has been done by the BBO team @ GERAD for the Hyperparameter Optimization (HPO) project in the context of the Alliance with *Huawei-Canada* between september 2023 and february 2024.

## Get started

The PDF document `approach.pdf` in the `Docs` folder thoroughly describes the theory behind our work, as well as our approach. It is recommended to read it first in order to fully understand what is undertaken here.

In the following, the reader is assumed to be familiar with the theory developed in this document, especially with: the Transformers architecture [[Vas+17]](#Vas17), the concept of instruction-tuning, the LoRA fine-tuning method [[Hu+21]](#Hu21), 4 widespread NLP tests (MMLU, BBH, DROP and HumanEval) and the family of LLaMA language models. [[Touv+23]](#Touv23)

## Approach

We perform the fine-tuning of the 7B parameter variant of LLaMA 2 on a 53k-sized dataset of instructions with the LoRA fine-tuning method. Recall that the LoRA method relies on the following quasi-equality:

```math
\forall x\in\mathbb{R}^d,\quad W_0x+\Delta Wx\simeq W_0x+\frac{\alpha}{r}BAx\enspace.
```

with $`B\in\mathbb{R}^{k\times r}, A\in\mathbb{R}^{r\times d}`$ and $`\alpha\in\mathbb{N}`$.

We seek to optimize the choice of 4 hyperparameters (HPs) within this context: $`r`$, $`\alpha`$, the dropout probability of the optimizer AdamW and the learning rate of the fine-tuning. Each combination is denoted as a vector of hyperparameters $`\theta=(r,\alpha,dropout,lr)^ \top`$.

## Requirements

The HuggingFace API is central to our experiment. It implements language models, training and test procedures. See [Transformers](https://huggingface.co/docs/transformers/index) and [PEFT](https://huggingface.co/docs/peft/index) especially.

Global requirements are:
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

* `bbo` contains all files needed to reproduce the experiment described in [experiment 1](#exp1).
* `bbo2` contains all files needed to reproduce the experiment described in [experiment 2](#exp2).
* `bbo3` contains all files needed to reproduce the experiment described in [experiment 3](#exp2).
* `blind_eval` contains data and scripts we used to generate text answers from our models in order to conduct the survey for human evaluation.
* `data` contains the data used for training and valiation.
* `eval` contains some scripts useful to run the evaluation of the model on a dataset.
* `nni` contains the files needed to reproduce an experiment that is described in deeper details in the appropriate folder.
* `plot` contains a script to draw a parallel plot from a statistics file.
* `train` contains scripts used to run the training phase of our pipeline.

## Pipeline implementation

The pipeline is usually broken down into 2 files:
1. `bb.py`
    * reads the encoded values of the 4 HPs given by NOMAD,
    * if a history file is provided, checks whether a very close point has already been evaluated. If so, returns the associated blackbox value,
    * calls ` eval.py` to perform the training and evaluation phases,
    * reads the results of the evaluation in a file and returns it to NOMAD.
2. `eval.py`
    * reads the encoded values of the 4 HPs given by NOMAD and translates them into actual values (see encoding for every experiment),
    * chooses the GPUs that will be used for computation (variable `cuda_visible_devices` at the beginning of the file). Before lauching an experiment, check which GPUs are available with `nvidia-smi` . A script like [nvidia-htop](https://github.com/peci1/nvidia-htop) can be useful if you need to see who is running processes on GPUs (and know how you can share the resources);
    * runs the following command in order to perform training and validation with the appropriate HPs. Elements that should be adapted to your local setup or customized are between braces `<the element>`.

```bash
source <venv_path>; export HF_HOME=<hf_path>; export WANDB_MODE=offline; 
CUDA_VISIBLE_DEVICES=<gpus> torchrun -r 2
--log_dir <log_path> --proc_per_node <nb_gpus> train/train_with_LoRa_mixed_data.py --model_name_or_path <pt_model> --data_path_train <training_data> --do_eval <eval> --data_path_eval <eval_data> --bf16 True --output_dir <checkpoints_path> --num_train_epochs <epochs> --per_device_train_batch_size <training_batch_size> --per_device_eval_batch_size <eval_batch_size> --gradient_accumulation_steps 8 --evaluation_strategy <eval_strategy> --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate <lr> --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --lora_rank <rank> --lora_dropout <dropout> --lora_alpha <alpha>; deactivate
```

this command line will:
* load the Python virtual environment in `venv_path` (should point to the `activate` file);
* indicate to the HuggingFace library where the language models have been downloaded (`hf_path`);
* run the fine-tuning as follows, by relying on the `train//train_with_LoRa_mixed_data.py` script:
    * use the GPUs listed in `gpus` (following the format: `0,1,2,3` for instance). `nb_gpus` should equal the amount of these GPUs;
    * use the pretrained model denoted in `pt_model` (use the name as displayed on the HuggingFace Hub);
    * all hyperparameter values are given here: `rank`, `alpha`, `dropout`, `lr`;
    * use the training data given in `training_data`;
    * use `epochs` training epochs with a batch size of `training_batch_size`;
    * if `eval` is set to `True`:
        * perform evaluation of the model on the dataset given in `eval_data` with a batch size of `eval_batch_size`;
        * the evaluation will be performed periodically depending on the value of `eval_strategy` (`"step"` at every training step (highly useless to perform it so often), `"epoch"` at every training epoch);
* the outputs will be stored as follows:
    * the logs from PyTorch will be saved in `log_path`;
    * the LoRA weights output from the training (checkpoints) will be saved in `checkpoints_path`.

Feel free to change some parameters as you wish.

## NOMAD

Giving `$python3 bb.py` as the blackbox to NOMAD will suffice to run this pipeline. As NOMAD does not handle natively some of the types we used to define the possible values for each HP, we encoded them and translated them in the Python script. Each encoding will be described in the appropriate section below.

## Already run experiments

Each experiment sets a specific objective function and uses specific sets of values for each variable.

### Experiment 1
<a id="exp1"></a>
This experiment uses the *MMLU score* of the language model as the objective function to *maximize*. It uses the whole Alpaca dataset. Possible values and encodings for each HP are as follows:

| HP | Possible values | NOMAD type | NOMAD encoding
|---|---|---|---|
|$`r`$|$`\{4,8,16,32,64,128\}`$|int|$`\{1,2,3,4,5,6\}`$|
|dropout|$`\{0,10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1\}`$|int|$`\{1,2,3,4,5,6\}`$|
|$`\alpha`$|$`[\![1,64]\!]`$|int|no need to encode|
|learning rate|$`[10^{-6}, 10^{-3}]`$|float|$`\log_{10}(lr)`$, so that NOMAD can choose values in $`[-6,-3]`$|


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