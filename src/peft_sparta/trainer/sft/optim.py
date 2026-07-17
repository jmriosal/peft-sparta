import torch
import math


def build_optimizer(model, config, task):
    
    params_to_optimize = model.parameters()

    weight_decay = config['weight_decay']

    decay, nodecay, optim_groups = [], [], []

    if weight_decay == 0.0:

        nodecay = params_to_optimize

    else:
        if config['peft_method'] == 'full_sft':

            # don't decay: bias & layernorm vectors and embedding matrices
            embedding_matrices = [m.weight for m in model.modules()
                                  if isinstance(m, torch.nn.Embedding)]
            for param in params_to_optimize:
                if param.dim() == 1 or any(param is m for m in embedding_matrices):
                    nodecay.append(param)
                else:
                    decay.append(param)

        elif config['peft_method'] in ['sparse', 'lora'] and task == 'SEQ_CLS':

            # don't decay: seq classification head
            head = (model.deltas['score.weight'] # model.train_params['deltas']['score.weight']
                    if config['peft_method'] == 'sparse' else
                    model.score.weight)
            for param in params_to_optimize:
                if param is head:
                    nodecay.append(param)
                else:
                    decay.append(param)

        else: # SEQ_CLS + head_only or CAUSAL_LM + sparse/lora

            decay = params_to_optimize

    if decay:
        optim_groups.append({'params': decay, 'weight_decay': weight_decay})
    if nodecay:
        optim_groups.append({'params': nodecay, 'weight_decay': 0.0})

    return torch.optim.AdamW(optim_groups,
                             lr=config['lr'],
                             betas=(0.9, 0.99),
                             fused=torch.cuda.is_available())


def build_lr_scheduler(optimizer, config, steps_per_epoch):
    assert config['lr_scheduler_type'] in ['linear', 'cosine']
    lr_max, lr_min = config['lr'], config['lr_min']
    warmup_steps = config['warmup_steps']
    decay_steps = steps_per_epoch * config['num_epochs'] - warmup_steps
    assert lr_max >= lr_min and decay_steps > 0
    def lr_lambda(i):
        """lr decay with warmup"""
        if lr_max == lr_min:
            return 1.
        elif i <= warmup_steps:
            x = i / warmup_steps # 0 -> 1
            return (lr_min + (lr_max - lr_min) * x) / lr_max
        elif i <= warmup_steps + decay_steps:
            x = (i - warmup_steps) / decay_steps # 0 -> 1
            if config['lr_scheduler_type'] == 'linear':
                x = 1 - x # 1 -> 0
            else: # 'cosine'
                x = 0.5 * (1 + math.cos(math.pi * x)) # 1 -> 0 
            return (lr_min + (lr_max - lr_min) * x) / lr_max
        else: # for extra train epochs beyond config['num_epochs']
            return lr_min / lr_max
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
