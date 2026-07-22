# SFT Trainer

`peft_sparta` includes a simple supervised fine-tuning trainer supporting the following SFT methods:

* SpaRTA
* LoRA
* Full Fine-Tuning

for both **sequence-classification (SEQ_CLS)** and **generation (CAUSAL_LM)** tasks.


## Install

The trainer needs extra dependencies:

```bash
pip install peft-sparta[trainer]

```

This adds them on top of the core install.


## Quickstart

### SEQ_CLS (sequence-classification) task

 ```python

from peft_sparta.trainer import SFT, SFT_Config
from datasets import load_dataset


# 1. Load a SEQ_CLS task dataset (see formats below)

ds = load_dataset('imdb')
train_ds = ds['train']
val_ds   = ds['test']


# 2. Choose and config the model to load and adapt for the SEQ_CLS task

model_config = {
    'task': 'SEQ_CLS',
    'name_or_fpath': 'google/gemma-2b',
    'num_classes': 2,
    'id2label': {0: 'negative', 1: 'positive'},
    'head_init': 'random',
    'new_tokens': None,
    'dtype': 'bfloat16',
}


# 3. Pick PEFT training method and configure it

peft_method = 'lora'

peft_config = {
    'task_type': 'SEQ_CLS', 
    'inference_mode': False, 
    'r': 16, 
    'lora_alpha': 16
    }

# 4. Setup SFT trainer (see defaults in SFT_Config)

sft_config = SFT_Config()

sft_config['num_epochs']  = 1
sft_config['lr']          = 1e-4
sft_config['bs_train']    = 24
sft_config['output_dir']  = 'runs/exp1'        # logs / plots (required)
sft_config['save_dir']    = 'runs/exp1/model'  # saved model  (required)

# optionally: config.update_from_yaml('my_config.yaml')

sft_trainer = SFT(
    sft_config,
    model_config,
    train_ds,
    val_ds,
    peft_method,
    peft_config,
  )

# 5. Train

sft_trainer.train()

# 6. Save

sft_trainer.plot_stats()   # saves plots into PDFs to output_dir
sft_trainer.save_stats()   # saves TensorBoard logs in output_dir
sft_trainer.save_model()   # saves adapter to save_dir

```


#### SEQ_CLS head initialization

For `SEQ_CLS` tasks, `model_config` controls how the classification head of the model is initialized via `head_init`.

- `head_init: 'random'`: the classification head is randomly initialized (and rescaled so classes initially have near-equal probabilities).

- `head_init: 'from_pretrained'`: initialize the classification head from the pre-trained model's **vocabulary head**, using a corresponding token per class. Only for **instruction-tuned** models (which have a usable vocab head).
Provide `response_classes`: a list of length `num_classes` giving the word the model would produce (after the instruction) for each class, **in label-index order** (matching `id2label`). The first token of each response word is used. For example:


```python
model_config = {
    'task': 'SEQ_CLS',
    'name_or_fpath': 'google/gemma-2b-it',   # instruct model
    'num_classes': 2,
    'id2label': {0: 'negative', 1: 'positive'},
    'head_init': 'from_pretrained',
    'response_classes': ['negative', 'positive'],   # 0 -> 'negative', 1 -> 'positive'
  }
```



### CAUSAL_LM (generation) task


```python

from peft_sparta.trainer import SFT, SFT_Config
from datasets import load_dataset

# 1. Load a CAUSAL_LM task dataset (see formats below)

ds = load_dataset('trl-lib/Capybara')['train']   # has a 'messages' column (chat format)
train_ds, val_ds = ds.train_test_split(test_size=1000, shuffle=True).values()

# 2. Choose model to load and adapt to the CAUSAL_LM task

model_config = {
    'task': 'CAUSAL_LM',
    'name_or_fpath': 'google/gemma-2b-it',   # instruct model with chat template
    'new_tokens': None,
    'dtype': 'bfloat16',
}

# 3. PEFT method + config

peft_method = 'lora'
peft_config = {
    'task_type': 'CAUSAL_LM',
    'inference_mode': False,
    'r': 16,
    'lora_alpha': 16,
}

# 4. Setup SFT trainer config

sft_config = SFT_Config()
sft_config['output_dir'] = 'runs/exp1'
sft_config['save_dir']   = 'runs/exp1/model'

sft_trainer = SFT(
    sft_config,
    model_config,
    train_ds,
    val_ds,
    peft_method,
    peft_config,
)

# 5. Train

sft_trainer.train()

# 6. Save

sft_trainer.save_model(merged=True)

```

Pick the pre-trained model to be compatible with your dataset format:
use an instruction-tuned model for the `messages` (chat) format;
a base model is fine for datasets with `prompt+completion` or plain `text` formats.


## Supported dataset formats

The trainer auto-detects the dataset format from the dataset's columns.

### SEQ_CLS 

- `text` + `label` columns.


### CAUSAL_LM

- `messages` column. Example messages must be in chat format for instruction tuning, i.e., a list of {"role", "content"} dicts, ending with an assistant turn. Loss is computed on the assistant's final response only. Use only with instruction-tuned models. The chat template is automatically applied during tokenization. These are typically referred to as single-turn conversational (chat) prompt-completion datasets.

- `prompt` + `completion` columns. Loss computed on the completion only. No chat template used.

- `text` column. Plain language modeling; loss computed on all tokens.


## PEFT methods

We provide support for the following training methods by choosing `peft_method` to be

- `'sparse'` for SpaRTA. Use `peft_config = {'sparsity': 0.999}` to select the level of sparsity, in this case 99.9%; you can also use e.g. `peft_config['frozen_modules'] = ['embed_tokens', 'mlp', 'norm', 'self_attn.o']` to completely freeze specific model modules.
- `'lora'` for LoRA (uses HF PEFT). Make sure to add a `task_type` in `peft_config`.
- `'head_only'` to train only the classification head (SEQ_CLS tasks only).
- `'full_sft'` (default) for full parameter fine-tuning (FFT).

Except for Full Fine-Tuning (FFT), if you add new tokens, `peft_config` must also include `train_new_tokens` (True/False).


## SFT_Config

`SFT_Config` is a dict with SFT trainer defaults; you can override any key before constructing SFT.


## Saving

```python
trainer.save_model(merged=False, save_tokenizer=True, sft_info=True, overwrite=False)
```
- `merged=True`: saves a standalone model with the adapter merged into the model (loads as a plain model, best for single-model serving and generation). For LoRA, this is an in-place merge that destroys the adapter: do not continue training after a merged save. For SpaRTA, the merge is non-destructive; you can keep training.

- `merged=False` (default): saves only the adapter. Best for merging adapters after training them, or saving disk space.

- `overwrite=False`: raises if `save_dir` already exists.

