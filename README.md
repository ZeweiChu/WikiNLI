# WikiNLI

The dataset and code for the paper [Mining Knowledge for Natural Language Inference from Wikipedia Categories](https://arxiv.org/abs/2010.01239).

## Dataset
The training and development dataset are under [data/WikiNLI](/data/WikiNLI)

In the paper we sampled 100k instances for our experiments, and the 100k version of training set can be found under [data/WikiNLI/100k](/data/WikiNLI/100k)

### Other languages

- WikiNLI constructed from Wikipedia of other languages are under [data/mWikiNLI](data/mWikiNLI), we provide four versions, Chinese(zh), French(fr), Arabic(ar) and Urdu(ur)
- WikiNLI constructed by translating English WikiNLI to other languages are under [data/trWikiNLI](data/trWikiNLI), we provide four versions, Chinese(zh), French(fr), Arabic(ar) and Urdu(ur)

## WordNet and WikiData

- The [WordNet](/data/wordnet) and [WikiData](/data/wikidata) we used in the experiments described in the WikiNLI paper

## Code

To pretrain a transformer based model on WikiNLI with the Huggingface transformers framework, use the following scripts. 

```bash
python code/run_wikinli.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased \
    --task_name wikinli \
    --num_train_examples 500000 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir ./data \
    --max_seq_length 40 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-5 \
    --num_train_epochs 1.0 \
    --save_steps 3000 \
    --logging_steps 3000 \
    --eval_all_checkpoints \
    --output_dir ./saved_outputs/bert-large
```

```bash
python code/run_wikinli.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name wikinli \
    --num_train_examples 500000 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir ./data \
    --max_seq_length 40 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 1e-5 \
    --warmup_steps 1000 \
    --num_train_epochs 3.0 \
    --save_steps 3000 \
    --logging_steps 3000 \
    --eval_all_checkpoints \
    --output_dir ./saved_outputs/roberta-large 
```

after the model is finished pretraining, modify the saved model by removing the top linear layer. 

```bash
mv [PATH]/pytorch_model.bin [PATH]/raw_pytorch_model.bin
python code/modify_saved_model.py [PATH]/raw_pytorch_model.bin [PATH]/pytorch_model.bin
``` 

A WikiNLI pretrained roberta-large model can be downloaded from [https://drive.google.com/file/d/1RJgewj2TPXI2lDNuxEO1gq9aSTkdRxiZ/view?usp=sharing](https://drive.google.com/file/d/1RJgewj2TPXI2lDNuxEO1gq9aSTkdRxiZ/view?usp=sharing)

To evaluate on NLI related tasks after pretraining on WikiNLI, follow the instructions of [evaluating GLUE tasks by Huggingface](https://github.com/huggingface/transformers/tree/master/examples/text-classification). 

### Dependency

- PyTorch 1.4.0
- transformers 3.1.0


## Citation
```
@inproceedings{chen2020mining,
      title={Mining Knowledge for Natural Language Inference from Wikipedia Categories}, 
      author={Mingda Chen and Zewei Chu and Karl Stratos and Kevin Gimpel},
      booktitle = {Findings of {EMNLP}},
      year={2020},
}
```
