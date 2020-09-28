# WikiNLI

dataset and code for the paper [Mining Knowledge for Natural Language Inference from Wikipedia Categories]()

## Dataset
The training and development dataset are under [data](/data)

## Code

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


### Dependency

- PyTorch 1.4.0
- transformers 3.1.0


## Citation
