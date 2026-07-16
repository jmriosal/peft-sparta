import os
import yaml
import math
import torch
from .data_processor import DataProcessor
from ...metrics import mcc, f1_score


PEFT_METHODS = ['lora', 'sparse', 'head_only', 'full_sft']

class SFT_Config(dict):
    def __init__(self):
        super().__init__()
        self['train_input_maxlen'] = 256
        self['bs_train']           = 24        # batch size of train dataloader (multiple of 8)
        self['bs_eval']            = 48        # batch size of val/test dataloader (multiple of 8)
        self['num_epochs']         = 1
        self['lr']                 = 1e-4      # learning rate (max) for the Adam optimizer
        self['lr_min']             = 1e-5      # lr_scheduler: decays lr from lr to lr_min
        self['lr_scheduler_type']  = 'cosine'  #               with a 'linear' or 'cosine' schedule
        self['warmup_steps']       = 100       #               after warmup
        self['early_stopping']     = False     # early stopping (ES)
        self['es_patience']        = 1         # ES: evaluations without improvement before stopping
        self['es_min_delta']       = math.inf  # ES: save model if improvement > es_min_delta (inf = never, 0 = any)
        self['attention_dropout']  = 0.0       # dropout rate for attention scores of transformer model
        self['adapter_dropout']    = 0.0       # dropout rate for adapter (SpaRTA, LoRA), if adapter is used
        self['weight_decay']       = 0.0       # (decoupled) weight decay regularization for the AdamW optimizer
        self['max_grad_norm']      = None      # gradient clipping
        self['amp']                = True      # automatic mixed precision (bfloat16)
        self['logging_freq']       = 100       # how often to log training progress: printouts + val eval
        self['output_dir']         = None
        self['save_dir']           = None

    def update_from_yaml(self, fpath):
        with open(fpath, 'r') as f:
            updates = yaml.safe_load(f)
            self.update(updates)


class SFT:
    def __init__(self,
                 config,
                 model_config,
                 train_dataset,
                 val_dataset=None,
                 peft_method=None,
                 peft_config=None):

        assert config['lr'] > 0, "lr must be positive"
        assert config['lr_min'] > 0 and config['lr_min'] <= config['lr'], "lr_min must be in (0, lr]"
        assert config['lr_scheduler_type'] in ('linear', 'cosine'), "lr_scheduler_type must be 'linear' or 'cosine'"
        assert config['es_patience'] >= 1, "Early stopping es_patience must be >= 1"
        assert config['es_min_delta'] >= 0.0, "Early stopping es_min_delta must be non-negative"
        assert config['logging_freq'] >= 1, "logging_freq must be >= 1"
        assert config['output_dir'], "Specify an output directory"
        assert config['save_dir'], "Specify a directory to save the model"

        peft_method = 'full_sft' if peft_method is None else peft_method.lower()
        if peft_method not in PEFT_METHODS:
            raise ValueError(f"peft_method = '{peft_method}' not supported")

        if peft_method not in ['lora', 'sparse']:
            del config['adapter_dropout']

        config['peft_method'] = peft_method
        self.config = config

        self.task = model_config.pop('task')
        if self.task == 'SEQ_CLS':
            from .model_loader import load_classification_model
            tokenizer, model = load_classification_model(model_config['name_or_fpath'],
                                                         model_config['num_classes'],
                                                         model_config['id2label'],
                                                         model_config['head_init'],
                                                         model_config['response_classes'], # instruct models
                                                         model_config['new_tokens'],
                                                         dtype=model_config.get('dtype', None),
                                                         attention_dropout=self.config['attention_dropout'])
        elif self.task == 'CAUSAL_LM':
            if peft_method == 'head_only':
                raise ValueError(f"{peft_method=} not supported for GENERATION tasks")
            from .model_loader import load_generative_model
            tokenizer, model = load_generative_model(model_config['name_or_fpath'],
                                                     model_config['new_tokens'],
                                                     dtype=model_config.get('dtype', None),
                                                     attention_dropout=self.config['attention_dropout'])
        else:
            raise NotImplementedError(f"Task: {self.task} not implemented")
  
        self.tokenizer = tokenizer

        self.num_added_tokens = len(model_config['new_tokens'] or [])
        if model_config['new_tokens'] and peft_config:
            if 'train_new_tokens' not in peft_config:
                raise ValueError("'peft_config' must include 'train_new_tokens' (True/False) "
                                 "when new tokens are added to the model")
            if peft_config.pop('train_new_tokens'):
                print(f">> Making fully-trainable the embeddings of {model_config['new_tokens'] = }")
                # make new token embeddings fully-trainable (in non full_sft methods)
                new_token_ids = self.tokenizer.convert_tokens_to_ids(model_config['new_tokens'])
                if peft_method == 'sparse':
                    peft_config['trainable_tokens'] = new_token_ids
                elif peft_method == 'lora': # https://github.com/huggingface/peft/pull/2376
                    peft_config['trainable_token_indices'] = {'embed_tokens': new_token_ids}
                elif peft_method == 'head_only':
                    peft_config['train_embeddings'] = True 

        if self.config['peft_method'] == 'sparse':
            print('>> Creating SpaRTA adapter')
            from ...sparta import SpaRTA # sparsify
            peft_config['dropout'] = self.config['adapter_dropout'] # sparta_dropout
            model = SpaRTA(model, **peft_config)
            
        elif self.config['peft_method'] == 'lora':
            print('>> Creating LoRA adapter')
            from peft import LoraConfig, get_peft_model
            peft_config['lora_dropout'] = self.config['adapter_dropout'] # 0.1
            lora_config = LoraConfig(**peft_config)
            model = get_peft_model(model, lora_config)
        
        elif self.config['peft_method'] == 'head_only':
            print('>> Making head only trainable parameters in PT model')
            from .head_only import HeadOnly
            model = HeadOnly(model, **peft_config)

        else:
            assert self.config['peft_method'] == 'full_sft'
            assert peft_config is None

        self.model = model # PT model
        
        # model placement decision
        if hasattr(self.model, 'hf_device_map'):
            # multi-gpu pipeline model already placed across devices
            self.device = self.model.device  # where input batches enter the model
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.device = device

        params_to_optimize = self.model.parameters()

        self.optimizer = self.configure_optimizer(params_to_optimize)

        self.data_processor = DataProcessor(self.tokenizer, self.task)

        self.train_dataloader = self.data_processor.build_dataloader(
            train_dataset,
            self.config['bs_train'],
            shuffle = True,
            max_token_len = self.config['train_input_maxlen'],
        )

        self.lr_scheduler = self.configure_lr_scheduler(self.optimizer)

        self.val_dataloader = self.data_processor.build_dataloader(
            val_dataset,
            self.config['bs_eval'],
        )

        self.stats = {'train_loss': [], 'val_loss': [], 'val_step': [], 'grad_norm': [], 'lr': []}
        if self.task == 'SEQ_CLS':
            self.stats['val_acc'] = []

        self.print_init_info()


    def print_init_info(self):

        print('>> Model:\n', self.model)
        if hasattr(self.model, 'hf_device_map'):
            print('device_map:', self.model.hf_device_map)
        if self.config['peft_method'] == 'full_sft':
            print(f"Num trainable parameters: {self.model.num_parameters():,d}")
        else: # lora, sparse, head_only
            self.model.print_trainable_parameters()

        print('>> Created tokenized dataloaders from datasets')
        if self.config['train_input_maxlen']:
            print(f'Num. of training examples after filtering: {len(self.train_dataloader.dataset)}')
        max_input_len = max(len(input_ids) for input_ids in self.train_dataloader.dataset['input_ids'])
        print(f"Max input token length when training: {max_input_len}")


    def configure_optimizer(self, params_to_optimize):

        weight_decay = self.config['weight_decay']

        decay, nodecay, optim_groups = [], [], []

        if weight_decay == 0.0:

            nodecay = params_to_optimize

        else:
            if self.config['peft_method'] == 'full_sft':

                # don't decay: bias & layernorm vectors and embedding matrices
                embedding_matrices = [m.weight for m in self.model.modules()
                                      if isinstance(m, torch.nn.Embedding)]
                for param in params_to_optimize:
                    if param.dim() == 1 or any(param is m for m in embedding_matrices):
                        nodecay.append(param)
                    else:
                        decay.append(param)

            elif self.config['peft_method'] in ['sparse', 'lora'] and self.task == 'SEQ_CLS':

                # don't decay: seq classification head
                head = (self.model.deltas['score.weight'] # self.model.train_params['deltas']['score.weight']
                        if self.config['peft_method'] == 'sparse' else
                        self.model.score.weight)
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
                                 lr=self.config['lr'],
                                 betas=(0.9, 0.99),
                                 fused=torch.cuda.is_available())


    def configure_lr_scheduler(self, optimizer):
        assert self.config['lr_scheduler_type'] in ['linear', 'cosine']
        lr_max, lr_min = self.config['lr'], self.config['lr_min']
        warmup_steps = self.config['warmup_steps']
        decay_steps = len(self.train_dataloader) * self.config['num_epochs'] - warmup_steps
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
                if self.config['lr_scheduler_type'] == 'linear':
                    x = 1 - x # 1 -> 0
                else: # 'cosine'
                    x = 0.5 * (1 + math.cos(math.pi * x)) # 1 -> 0 
                return (lr_min + (lr_max - lr_min) * x) / lr_max
            else: # for extra train epochs beyond config['num_epochs']
                return lr_min / lr_max
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    def train(self, num_epochs=None):

        num_epochs = self.config['num_epochs'] if num_epochs is None else num_epochs

        self.training_progress(setup=True)

        self.model.train()

        for epoch in range(num_epochs):
            for i, batch in enumerate(self.train_dataloader):

                batch = {k: v.to(self.device) for k, v in batch.items()} # input_ids, attention_mask, labels

                # forward pass
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.config['amp']):
                    outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                self.stats['train_loss'].append(loss.item())

                # backward pass  
                self.optimizer.zero_grad()
                loss.backward()

                # clip params grad (if too large) 
                if self.config['max_grad_norm']:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        (param for group in self.optimizer.param_groups for param in group['params']), 
                        self.config['max_grad_norm'], norm_type=2).item()
                    self.stats['grad_norm'].append(grad_norm)
                
                # params update
                self.optimizer.step()
                self.stats['lr'].append(self.lr_scheduler.get_last_lr()[0])
                self.lr_scheduler.step()

                # logging
                stop = self.training_progress(epoch, i, num_epochs)

                if stop:
                    print('\n*** stopping training early ***\n')
                    break
            else:
                continue
            break


    def training_progress(self, epoch=None, i=None, num_epochs=None, *, setup=False):

        freq = self.config['logging_freq']

        num_batches = len(self.train_dataloader)
        width = len(str(num_batches)) # 'Batch' field width for printouts

        is_last = not setup and epoch == num_epochs - 1 and i == num_batches - 1

        if setup: # before training starts
            header = f"{'Epoch':^5}  {'Batch':^{2*width+1}}  {'Train Loss':>10}"
            if self.val_dataloader:
                header += f"  {'Val Loss':>10}"
                if self.task == 'SEQ_CLS':
                    header += f"  {'Val Acc':>8}"
            print(header)
            if not self.val_dataloader or len(self.stats['val_loss']) > 0:
                return False
            line = f"{'-':^5}  {'-':^{2*width+1}}  {'-':>10}"
            if self.config['early_stopping']:
                self.best_saved = math.inf  # track best saved val loss

        elif (epoch * num_batches + i + 1) % freq and not is_last: # skip (no printouts/eval/logging)
            return False

        else: # during training
            sample_size = 1000 # num of training examples
            target_window = math.ceil(sample_size/self.train_dataloader.batch_size) # num of batches
            window = min(target_window, freq)  # window <= freq
            last_losses = self.stats['train_loss'][-window:]
            train_loss = sum(last_losses)/len(last_losses)  # averaged over most recent window
            line = f"{epoch+1:^5d}  {i+1:{width}d}|{num_batches:{width}d}  {train_loss:10.4f}"

        if self.val_dataloader:
            results = self.evaluate(self.val_dataloader)
            self.stats['val_loss'].append(results['loss'])
            self.stats['val_step'].append(len(self.stats['train_loss']))
            line += f"  {results['loss']:10.4f}"
            if self.task == 'SEQ_CLS':
                self.stats['val_acc'].append(results['accuracy'])
                line += f"  {results['accuracy']*100:7.1f}%"

            if self.config['early_stopping']:
                n = len(self.stats['val_loss'])
                if n > 2 and epoch > 0:
                    new = self.stats['val_loss'][-1]
                    if new < self.best_saved - self.config['es_min_delta']: # improvement
                        self.save_model(merged=False, save_tokenizer=False, sft_info=False, overwrite=True)
                        self.best_saved = new
                    else:
                        best = min(self.stats['val_loss'])
                        patience = n - 1 - self.stats['val_loss'].index(best)
                        if patience >= self.config['es_patience']:
                            print(line)
                            return True # stop training

            self.model.train()

        print(line)
        return False


    @torch.no_grad()
    def evaluate(self, eval_dataloader, verbose=False):
        self.model.eval()        
        eval_loss = 0.0
        if self.task == 'SEQ_CLS':
            accuracy = 0.0
            if verbose:
                cm = torch.zeros((self.model.num_labels, self.model.num_labels), dtype=torch.int32)
        for batch in eval_dataloader: 
            batch = {k: v.to(self.device) for k, v in batch.items()} # input_ids, attention_mask, labels
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.config['amp']):
                outputs = self.model(**batch, use_cache=False)
            eval_loss += outputs.loss.item()
            if self.task == 'SEQ_CLS':
                predictions = torch.argmax(outputs.logits, dim=-1)
                accuracy += (predictions == batch['labels']).sum().item()
                if verbose:
                    for label, pred in zip(batch['labels'].tolist(), predictions.tolist()):
                        cm[label, pred] += 1 # confusion matrix
        eval_loss /= len(eval_dataloader)
        if self.task == 'SEQ_CLS':
            accuracy /= len(eval_dataloader.dataset)

        if verbose:
            print(f'Evaluation Loss: {eval_loss:.3f}')
            if self.task == 'SEQ_CLS':
                print(f'Accuracy: {accuracy*100:.1f}%')
                print(f'Confusion Matrix:\n', cm/cm.sum())
                print('Accuracy by class (row):')
                acc_by_class = torch.diag(cm) / torch.sum(cm, dim=1)
                class_labels = eval_dataloader.dataset.features['labels'].names
                for l, acc in zip(class_labels, acc_by_class):
                    print(f" - '{l}': {acc.item()*100:.1f}%")
                bacc = acc_by_class.mean().item()
                print(f'Balanced accuracy: {bacc*100:.1f}%')
                if cm.shape == (2,2):
                    print(f'MCC: {mcc(cm):.3f}')
                    print(f'F1-score: {f1_score(cm):.3f}')
        else:
            results = {'loss': eval_loss}
            if self.task == 'SEQ_CLS':
                results['accuracy'] = accuracy
            return results  


    def save_model(self, merged=True, save_tokenizer=True, sft_info=True, overwrite=False):

        save_dir = self.config['save_dir']
        os.makedirs(save_dir, exist_ok=overwrite)

        if type(self.model).__name__ == 'SpaRTA':
            self.model.save(save_dir, merged)
            if not merged:
                if self.task == 'SEQ_CLS':
                    head = self.model.model.score
                    weight_init = self.model.get_init_param(head.weight)
                    torch.save(weight_init.data,
                               os.path.join(save_dir,'head_init.pt'))
                if self.num_added_tokens > 0:
                    embeddings = self.model.model.get_input_embeddings()
                    weight_init = self.model.get_init_param(embeddings.weight)
                    torch.save(weight_init.data[-self.num_added_tokens:].clone(),
                               os.path.join(save_dir,'new_embeddings_init.pt'))
        else:
            self.model.save_pretrained(save_dir)
        
        if save_tokenizer:
            self.tokenizer.save_pretrained(save_dir)      
        
        if sft_info:
            with open(os.path.join(save_dir, 'sft_info.txt'), 'w') as f: 
                f.write('config:\n')
                for k, v in self.config.items():
                    f.write(' %s: %s\n' % (k, v))
                #f.write('train_loss: '+ str(self.stats['train_loss'][-1]) +'\n')  
            
        print('\nModel saved in: %s' % save_dir)


    def save_stats(self, subdir=''):

        from torch.utils.tensorboard import SummaryWriter

        logdir = os.path.join(self.config['output_dir'], subdir)

        with SummaryWriter(logdir) as writer:
            for tag, ts in self.stats.items():
                for i, v in enumerate(ts):
                    writer.add_scalar(tag, v, i)

        print("\n>> To visualize 'stats' use:\n\n\t",
              f"tensorboard --logdir {logdir} --bind_all --port 60000\n")


    def plot_stats(self, fname_prefix=''):

        from . import plots

        if self.task == 'SEQ_CLS':
            num_classes = self.model.num_labels # self.train_dataloader.dataset.features['labels'].num_classes
            loss_baseline = -math.log(1 / num_classes)
        else:
            loss_baseline = None

        plots.plot_stats(
            self.stats, self.config, loss_baseline,
            fname_prefix=fname_prefix
        )


