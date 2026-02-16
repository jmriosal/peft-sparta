import os
import torch
from torch import nn 
import safetensors.torch


class SpaRTA(nn.Module):
    """
    Samples a sparse subset of model parameters and marks them as trainable while freezing the rest.

    Args:
        model (nn.Module): Pre-trained model to be adapted.
        sparsity (float): Target fraction of total model parameters to make non-trainable.
        frozen_modules (list[str], optional): Modules to make entirely frozen (non-trainable). 
            Classification heads ('score') will always be fully-trainable by default. 
        trainable_tokens (list[int], optional): Token ids whose embeddings should be fully-trainable. 
            Useful for newly added (special) tokens to the vocabulary. 
        dropout (float, optional): Dropout probability applied to the trainable parameters during training. 
    """

    def __init__(self,
                 model,
                 sparsity,
                 frozen_modules=['embed_tokens', 'self_attn.q', 'self_attn.k', 'mlp', 'norm'],
                 trainable_tokens=None,
                 dropout=0.0):

        super().__init__()
        
        assert 0 < sparsity < 1
        assert 'score' not in frozen_modules # classification head is fully-trainable 
        if trainable_tokens: # list of (unique) token ids
            assert len(trainable_tokens) == len(list(set(trainable_tokens)))
        assert 0 <= dropout < 1
        
        self.model = model
        if hasattr(model, 'get_input_embeddings'): 
            # transformers pretrained model
            self.embed_weight = model.get_input_embeddings().weight
        else:
            assert trainable_tokens is None
            self.embed_weight = None
        self.sparsity = sparsity
        self.frozen_modules = frozen_modules
        self.trainable_tokens = trainable_tokens

        nonfrozen_keep_prob = self.compute_keep_prob(sparsity)

        indices, deltas = {}, {}
        for name, param in model.named_parameters():
            # freeze (pre-trained) model params
            param.requires_grad = False
            
            # randomly select trainable (scalar) params
            if name == 'score.weight': # classification head
                keep_prob = 1.0 # fully-trainable
            elif any(k in name for k in self.frozen_modules):
                keep_prob = 0.0 # non-trainable
            else:
                keep_prob = nonfrozen_keep_prob
            indices[name] = self.sample_trainable(param, keep_prob)    
            deltas[name] = torch.zeros(indices[name].shape[0],
                                       dtype=torch.float32, # param.dtype
                                       device=param.device)

        self.indices = BufferDict(indices)
        self.deltas = ParameterDict(deltas)

        self.param_names = list(indices.keys())

        self.original_chosen_params = BufferDict(self.clone_chosen_params())

        self.train()

        if dropout > 0:
            self.dropout = torch.nn.Dropout(p=dropout, inplace=True).train()
        else:
            self.dropout = None

        if hasattr(model, 'num_labels'): # model for sequence classification
            self.num_labels = model.num_labels

        # for compatibility with transformers based trainers:
        #   SpaRTA model (self) must inherits some attributes 
        #   from input (PreTrainedModel) model
        required_attrs = ['config', 'add_model_tags']
        for attr_name in required_attrs:
            if hasattr(model, attr_name):
                setattr(self, attr_name, getattr(model, attr_name))

    def __repr__(self):
        return (f"SparseModel(sparsity={self.sparsity},"
                f" frozen_modules={self.frozen_modules},"
                f" dropout_rate={self.dropout.p if self.dropout else 0.0},"
                f"\n  {self.model}\n)")

    def compute_keep_prob(self, sparsity):
        total = sum(p.numel() for p in self.model.parameters())
        keep = (1 - sparsity) * total # as trainable
        if hasattr(self.model, 'score'):
            chead_size = self.model.score.weight.numel()
            if chead_size > keep:
                msg = (f"{sparsity=} is too high: "
                       f"Choose sparsity < {1-chead_size/total:.6f} " # keep > chead_size
                       f"to at least keep the classification head fully-trainable.")
                raise ValueError(msg)
            keep -= chead_size
            total -= chead_size
        else:
            chead_size = 0
        if self.trainable_tokens: 
            # size of fully-trainable tokens within embed matrix (num of params)
            ftt_size = len(self.trainable_tokens) * self.model.get_input_embeddings().embedding_dim
            if ftt_size > keep:
                threshold = 1 - (chead_size + ftt_size)/(total + chead_size)
                msg = (f"{sparsity=} is too high: "
                       f"Choose sparsity < {threshold:.6f} "
                       "to keep your chosen token embeddings fully-trainable.")
                raise ValueError(msg)
            keep -= ftt_size
            if any('embed' in m for m in self.frozen_modules):
                # frozen embeddings
                pass # to avoid double counting when removing frozen from total (below)
            else:
                # embeddings not frozen
                total -= ftt_size
        else:
            ftt_size = 0
        frozen = sum(p.numel() for name, p in self.model.named_parameters()
                     if any(k in name for k in self.frozen_modules))
        keep_prob = keep / (total - frozen) # select only from non-frozen param tensors
        if keep_prob >= 1.:
            if self.trainable_tokens and any('embed' in m for m in self.frozen_modules):
                threshold = (frozen - ftt_size) / (total + chead_size)
            else:
                threshold = frozen / (total + chead_size + ftt_size) 
            msg = (f"{sparsity=} is too low: "
                   f"Not enough non-frozen parameters to choose from during sparsification. "
                   f"Set a sparsity > {threshold:.6f} or reduce frozen layers.")
            raise ValueError(msg)
        return keep_prob

    def sample_trainable(self, param, keep_prob):        
        indices_dtype = torch.int32
        if keep_prob == 1.0:
            indices = torch.ones(param.shape).nonzero().to(indices_dtype)
        elif keep_prob == 0.0:
            if param is self.embed_weight and self.trainable_tokens:
                token_ids = torch.tensor(self.trainable_tokens)
                n, d = len(token_ids), param.shape[1]
                indices = torch.stack([token_ids.unsqueeze(1).repeat(1, d).view(-1),
                                       torch.arange(d).repeat(1, n).squeeze()], dim=1)
            else:
                indices = torch.empty((0, param.dim()), dtype=indices_dtype)
        else: # 0 < keep_prob < 1
            mask = param.bernoulli(keep_prob)
            if param is self.embed_weight and self.trainable_tokens:
                mask[self.trainable_tokens] = 1.
            indices = mask.nonzero().to(indices_dtype)
        return indices

    @torch.no_grad()
    def clone_chosen_params(self):
        cloned_params = {}
        for name, param in self.model.named_parameters():
            indices = self.indices[name].int().unbind(1)
            cloned_params[name] = param[indices].detach().clone() # same dtype & device
        return cloned_params

    def parameters(self): # only trainable
        return self.deltas.values() # iter

    def named_parameters(self): # only trainable
        return ((f"deltas.{name}", delta) for name, delta in self.deltas.items()) # iter

    def num_trainable_parameters(self, printout=True):
        n_trainable = sum(param.numel() for param in self.parameters())
        if printout:
            pct = n_trainable/self.model.num_parameters()*100
            print(f"Num trainable parameters: {n_trainable:,d} ({pct:.5f}%)")
        else:
            return n_trainable

    def merge_deltas(self):        
        # add sparse deltas
        for name, param in self.model.named_parameters():
            param.detach_() # remove op history from model (frozen) params
            indices = self.indices[name].int().unbind(1)
            w = self.original_chosen_params[name].float()
            d = self.deltas[name].float()
            if self.dropout and self.training:
                a = torch.ones_like(d)
                a = self.dropout(a)
                param[indices] = (w + a * d).to(dtype=param.dtype)
            else: # dropout.eval()
                param[indices] = (w + d).to(dtype=param.dtype)

    def forward(self, *args, **kwargs):
        if self.training:
            self.merge_deltas()
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        if self.training:
            self.eval()
        return self.model.generate(*args, **kwargs)
        
    @torch.no_grad()    
    def save(self, save_dir, merged=True):
        if merged:
            training = self.training
            self.eval() # merge_deltas with no_grad, if deltas not already merged
            self.model.save_pretrained(save_dir)
            if training:
                self.train()
            print('\nSparse model saved as a fully merged model in: %s' % save_dir)
        else:
            state_dict_to_save = {} 
            for k, v in self.state_dict().items():
                # only indices and deltas (adapter)
                names = k.split('.') 
                if names[0] == 'indices':
                    name = f"indices.{self.indices.find_key(names[1])}"
                elif names[0] == 'deltas':
                    name = f"deltas.{self.deltas.find_key(names[1])}"
                else: # model, original_chosen_params
                    continue
                state_dict_to_save[name] = v.contiguous()
            safetensors.torch.save_file(
                state_dict_to_save, 
                os.path.join(save_dir,'sparse_deltas.safetensors'))
            self.model.config.sparta_pretrained_model = self.model.config._name_or_path
            self.model.config.save_pretrained(save_dir) 
            print('\nModel sparse delta parameters saved in: %s' % save_dir)

    def train(self):
        self.model.train()
        self.training = True

    @torch.no_grad()
    def eval(self):
        if self.training:
            self.model.eval()
            self.training = False
            self.merge_deltas() # dropout.eval()



class BufferDict(nn.Module):
    def __init__(self, buf_dict=None):
        super().__init__()
        self.key_to_attr = {} 
        self.attr_to_key = {} 
        if buf_dict:
            for k, v in buf_dict.items():
                self[k] = v

    def __getitem__(self, key):
        if key in self.key_to_attr:
            attr = self.key_to_attr[key]
            return self.get_buffer(attr) # getattr(self, attr)
        raise KeyError(f"{key=} not found")

    def __setitem__(self, key, buffer):
        name = key.replace('.','__')
        self.key_to_attr[key] = name
        self.attr_to_key[name] = key
        self.register_buffer(name, buffer)

    def __contains__(self, key):
        return key in self.key_to_attr

    def __len__(self):
        return len(self.key_to_attr)

    def __iter__(self):
        return iter(self.key_to_attr.keys())

    def keys(self):
        return self.key_to_attr.keys()

    def values(self):
        return (self.get_buffer(buffer_name) for buffer_name in self.key_to_attr.values())
    
    def items(self):
        return ((key, self.get_buffer(buffer_name)) for key, buffer_name in self.key_to_attr.items())

    def find_key(self, buffer_name):
        return self.attr_to_key[buffer_name]



class ParameterDict(nn.Module):
    def __init__(self, param_dict=None):
        super().__init__()
        self.key_to_attr = {} 
        self.attr_to_key = {} 
        if param_dict:
            for k, v in param_dict.items():
                self[k] = v

    def __getitem__(self, key):
        if key in self.key_to_attr:
            attr = self.key_to_attr[key]
            return self.get_parameter(attr) # getattr(self, attr)
        raise KeyError(f"{key=} not found")

    def __setitem__(self, key, param):
        name = key.replace('.','__')
        self.key_to_attr[key] = name
        self.attr_to_key[name] = key
        setattr(self, name, nn.Parameter(param))

    def __contains__(self, key):
        return key in self.key_to_attr

    def __len__(self):
        return len(self.key_to_attr)

    def __iter__(self):
        return iter(self.key_to_attr)

    def keys(self):
        return self.key_to_attr.keys()

    def values(self):
        return (self.get_parameter(param_name) for param_name in self.key_to_attr.values())

    def items(self):
        return ((key, self.get_parameter(param_name)) for key, param_name in self.key_to_attr.items())

    def find_key(self, param_name):
        return self.attr_to_key[param_name] 
    
