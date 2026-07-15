import os
import json
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from safetensors import safe_open


if torch.cuda.device_count() > 1:
    _device_map = 'auto' # loads model into multiple gpus (pipeline parallelism)
else:
    _device_map = None   # loads model to cpu and SFT will move it to cuda


def load_generative_model(model_name, new_tokens=None, **kwargs):

    device_map = kwargs.pop('device_map', _device_map)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, **kwargs)

    ## Commented Out: it may interfere with accelerate
    #if hasattr(model, 'hf_device_map'):
    #    # move head outout (logits) to model.device (cuda:0) where labels are
    #    def set_output_device_hook(module, input, output):
    #        return output.to(model.device)
    #    lm_head = model.get_output_embeddings()
    #    lm_head.register_forward_hook(set_output_device_hook)

    # pad_token setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # used by collator (creates batches)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id  # model ignores pad_tokens when computing loss

    if new_tokens:
        add_tokens(tokenizer, new_tokens)
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


@torch.no_grad()
def load_classification_model(model_name,
                              num_classes,
                              id2label,
                              head_init='random',
                              response_classes=None, # instruction-tuned PT models only
                              new_tokens=None,
                              **kwargs):
    
    device_map = kwargs.pop('device_map', _device_map)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=num_classes,
                                                               id2label=id2label,
                                                               device_map=device_map,
                                                               **kwargs)

    model.score.weight.data = model.score.weight.data.to(dtype=torch.float32)
    model.score.register_forward_pre_hook(
        lambda module, input: (input[0].to(dtype=torch.float32),)
    )

    # classification head init params
    if head_init == 'random':
        # pre-initialized from N(0, std=0.02):
        #   logit std = sigma_h * sigma_W * sqrt(in_features), with sigma_W = 0.02
        # re-initializing by rescaling with scale / sqrt(in_features):
        #   logit std = sigma_h * 0.02 * scale
        # goal: logits near zero -> near-uniform, equal-prob predictions at init
        #   (but very small logits -> more grad steps needed to reach useful scale)
        scale = 3. # empirical: reduce for larger models
                   # since sigma_h tends to grow with model size despite layer norm
        model.score.weight.data *= scale / math.sqrt(model.score.in_features)

    
    elif head_init == 'from_pretrained':
        # reuse vocab head to init classification head (only for instruct-tuned PT models)
        vocab = tokenizer.get_vocab() # vocab[token] <=> tokenizer.convert_tokens_to_ids(token)
        assert len(response_classes) == num_classes
        print('model_responses:', response_classes)
        classification_tokens = [tokenizer.tokenize(res)[0] for res in response_classes]
        print('classification_tokens:', classification_tokens)
        class_ids = [vocab[token] for token in classification_tokens]
        assert len(set(class_ids)) == num_classes

        if model.config.tie_word_embeddings:  # tied
            embed_tokens = model.get_input_embeddings()
            w = embed_tokens.weight.data[class_ids].clone().detach()
        else:  # untied: head not in memory, read from disk
            head_fpath, head_param_name = find_head(model)
            with safe_open(head_fpath, framework='pt') as f:
                w = f.get_slice(head_param_name)[class_ids]
                                                                                
        model.score.weight.data = w.to(model.score.weight.data.device,
                                       model.score.weight.data.dtype)
    else:
        raise ValueError(f"head_init = '{head_init}', " +
                         "but only options are 'random' or 'from_pretrained'")


    if hasattr(model, 'hf_device_map'):
        # move head outout (logits) to model.device (cuda:0) where labels are
        def set_output_device_hook(module, input, output):
            return output.to(model.device)  
        model.score.register_forward_hook(set_output_device_hook)


    # pad_token setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    if new_tokens:
        num_added_tokens = add_tokens(tokenizer, new_tokens)
        resize_token_embeddings(model, num_added_tokens)

    return tokenizer, model



def find_head(model): # untied case
    
    if os.path.isdir(model.config._name_or_path):
        model_dir = model.config._name_or_path
    else:
        cache_dir = os.getenv('HF_HOME', '~/.cache/huggingface/')
        model_name = model.config._name_or_path.replace('/', '--')
        cache_dir = os.path.join(cache_dir, 'hub', f"models--{model_name}")
        if not os.path.isdir(cache_dir):
            raise ValueError(f"PT model '{model.config._name_or_path}' not in HF cache")
        with open(os.path.join(cache_dir, 'refs', 'main')) as f:
            revision = f.read().strip()
        model_dir = os.path.join(cache_dir, 'snapshots', revision)

    head_param_names = [  # most common ones
        'lm_head.weight',
        'embed_out.weight',
        'output_layer.weight'
    ]
    try: # params sharded in multiple files
        index_file = os.path.join(model_dir, 'model.safetensors.index.json')
        with open(index_file, 'r') as f:
            weight_map = json.load(f)['weight_map']
        for param_name in head_param_names:
            if param_name in weight_map:
                fname = weight_map[param_name]
                head_fpath = os.path.join(model_dir, fname)
                break
        else:
            raise ValueError(f"Could not find LM head in {index_file}.\n"
                             f"Tried with {head_param_names = }")
    except FileNotFoundError:
        fname = 'model.safetensors' # params stored in a single file
        head_fpath = os.path.join(model_dir, fname)
        with safe_open(head_fpath, framework="pt") as f:
            param_names = set(f.keys())
        for param_name in head_param_names:
            if param_name in param_names:
                break
        else:
            raise ValueError(f"Could not find LM head in {head_fpath}.\n"
                             f"Tried with {head_param_names = }")

    return head_fpath, param_name



def add_tokens(tokenizer, new_tokens):
    try:  # transformers version < 5.0
        assert all(token not in tokenizer.additional_special_tokens for token in new_tokens)
        n = tokenizer.add_special_tokens({'additional_special_tokens': new_tokens},
                                         replace_additional_special_tokens=False)
    except (AttributeError, TypeError):  # transformers version >= 5.0
        assert all(token not in tokenizer.all_special_tokens for token in new_tokens)
        n = tokenizer.add_special_tokens({'additional_special_tokens': new_tokens},
                                         replace_extra_special_tokens=False)
    assert n == len(new_tokens)
    print(f">> {n} new tokens added:")
    for token in new_tokens:
        print(f"   - Added token '{token}' with id {tokenizer.convert_tokens_to_ids(token)}")
    return n



@torch.no_grad()
def resize_token_embeddings(model, num_added_tokens=1, init_centered=False):
    embeddings = model.get_input_embeddings()
    new_embeddings = torch.randn(num_added_tokens, embeddings.embedding_dim,
                                 dtype = embeddings.weight.dtype, 
                                 device = embeddings.weight.device) 
    if not init_centered: 
        new_embeddings = new_embeddings * model.config.initializer_range
    else: # https://nlp.stanford.edu/~johnhew/vocab-expansion.html
        new_embeddings = new_embeddings * 1e-4 + torch.mean(embeddings.weight.data, axis=0)
    embeddings.weight.data = torch.cat((embeddings.weight.data, new_embeddings))
    embeddings.num_embeddings += num_added_tokens
    model.config.vocab_size += num_added_tokens
