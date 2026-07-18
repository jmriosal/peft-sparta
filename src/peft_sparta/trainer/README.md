

# SFT Trainer

`peft_sparta` includes a simple supervised fine-tuning trainer supporting 

* SpaRTA, 
* LoRA, 
* head-only, and 
* full fine-tuning, 

for both `sequence-classification` and `causal-LM` tasks.

## Install

The trainer needs extra dependencies:

```bash
pip install peft-sparta[trainer]

```

This adds them on top of the core install.

## Quickstart

 ```python

from peft_sparta.trainer import SFT, SFT_Config
from datasets import load_dataset


# 1. Task (SEQ_CLS) datasets (see formats below)
train_ds = load_dataset('imdb', split='train')
val_ds   = load_dataset('imdb', split='test')


# 2. Choose and config the model to load and adapt for SEQ_CLS task
model_config = {
    'task': 'SEQ_CLS',
    'name_or_fpath': 'google/gemma-2b',
    'num_classes': 2,
    'id2label': {0: 'negative', 1: 'positive'},
    'head_init': 'random',
    'new_tokens': None,
}


# 3. Pick PEFT training method and config
peft_method = 'lora'
peft_config = {
    'task_type': 'SEQ_CLS', 
    'inference_mode': False, 
    'r': 16, 
    'lora_alpha': 16
    }

# 4. Setup SFT trainer (see defaults in SFT_Config)

config = SFT_Config()
config['num_epochs']  = 1
config['lr']          = 1e-4
config['bs_train']    = 24
config['output_dir']  = 'runs/exp1'        # logs / plots (required)
config['save_dir']    = 'runs/exp1/model'  # saved model  (required)

# optionally: config.update_from_yaml('my_config.yaml')

trainer = SFT(
    config,
    model_config,
    train_ds,
    val_dataset=val_ds,
    peft_method=peft_method,
    peft_config=peft_config,
  )

# 5. Train

trainer.train()

# 6. Save + inspect

trainer.save_model()   # saves adapter to save_dir
trainer.plot_stats()   # saves plots into PDFs to output_dir
trainer.save_stats()   # saves TensorBoard logs in output_dir

```


## Supported dataset formats

The trainer auto-detects the dataset format from the dataset columns.

### SEQ_CLS 

- `text` + `label` columns.


### CAUSAL_LM

- `messages` column. Example messages must be in chat format for instruction tuning, i.e., a list of {"role", "content"} dicts and must end with an assistant turn. Loss is computed on the assistant response only. Use only with instruction-tuned models. The chat template is automatically applied during tokenization. These type of datasets are typically refer to as single-turn conversational prompt-completion datasets.

- `prompt` + `completion` columns. Loss computed on the completion only. No chat template used.

- `text` column. plain language modeling; loss computed on all tokens.

## PEFT methods (peft_method)

- `sparse` for SpaRTA
- `lora` for LoRA (uses HF PEFT). Make sure to ass a task_type in peft_config.
- `head_only`, to train only the classification head (SEQ_CLS tasks only)   
- `full_sft` (default) for full fine-tuning (FFT)

 
Except for FFT, if you add new tokens, peft_config must also include train_new_tokens (True/False).

## Saving

```python
trainer.save_model(merged=False, save_tokenizer=True, sft_info=True, overwrite=False)
```
- merged=True: writes a standalone model with the adapter merged into the base model (loads as a plain model). For LoRA this is an in-place merge that destroys the adapter — do not continue training after a merged save. For SpaRTA the merge is non-destructive; you can keep training.

- merged=False: saves the adapter only.

- overwrite=False raises if save_dir already exists.

  
## SFT_Config

`SFT_Config` is a dict with SFT trainer defaults; you can override any key before constructing SFT. 

