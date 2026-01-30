import os
import json
import torch
import safetensors.torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from .metrics import mcc, f1_score, bacc


class SpaRTAforSequenceClassification:
    """Loads a Sparse Adapted SeqClassification Model for Inference"""

    @torch.no_grad()
    def __init__(self, adapter, model=None, device='cpu', input_template=None):

        self.adapter = adapter
        
        # read PT model config 
        with open(os.path.join(adapter, 'config.json'), 'r') as f:
            config = json.load(f)

        # sequence classification task
        assert(config['problem_type'] == 'single_label_classification') 
        self.id2label = config['id2label']
        
        # get headless base model
        if model:
            # reuse loaded model
            assert(model.config._name_or_path == config['sparta_pretrained_model'])
            base_model = model.base_model
        else:
            # load from disk
            base_model = AutoModel.from_pretrained(config['sparta_pretrained_model'],
                                                   torch_dtype=torch.bfloat16)
        base_model.eval().to(device)
        hidden_size = base_model.config.hidden_size

        # add new token embeddings if added to model
        if any(f.startswith('new_embeddings_init.') for f in os.listdir(adapter)):
            embeddings = base_model.get_input_embeddings()
            new_embeddings_init = load_init_tensor(adapter, 'new_embeddings',
                                                   embeddings.weight.device
                                                  ).to(dtype=embeddings.weight.dtype)
            num_added_tokens = new_embeddings_init.shape[0]
            with torch.no_grad():
                embeddings.weight.data = torch.cat((embeddings.weight.data, new_embeddings_init))
            embeddings.num_embeddings += num_added_tokens
            base_model.config.vocab_size += num_added_tokens
        
        # merge SpaRTA adapter into base model
        sparse_deltas = load_sparse_deltas(adapter)
        for name, param in base_model.named_parameters():  
            name = base_model.base_model_prefix + '.' + name
            indices = sparse_deltas['indices'][name].to(device).unbind(1)
            # param[indices] += sparse_deltas['deltas'][name].to(device)
            chosen_params = param[indices].float()
            deltas = sparse_deltas['deltas'][name].to(device).float()
            param[indices] = (chosen_params + deltas).to(param.dtype)
        del indices

        # create sequence classification head
        w0 = load_init_tensor(adapter, 'head', device)
        num_labels = w0.shape[0]
        assert(num_labels == len(self.id2label))
        deltas = sparse_deltas['deltas']['score.weight'].detach().to(device)
        assert(len(deltas) // hidden_size == num_labels) # head fully trainable
        deltas = deltas.view(num_labels, hidden_size) 
        head = torch.nn.Linear(hidden_size, num_labels, bias=False).to(device)
        head.weight.data = (w0.float() + deltas.float()).to(dtype=head.weight.dtype) # w0 + deltas
        del w0, deltas, sparse_deltas
        
        self.base_model, self.head, self.device = base_model, head, device

        if os.path.isfile(os.path.join(adapter, 'tokenizer_config.json')):
            tk_source = adapter
        else:
            # if tokenizer is not modified during adaption by adding new (special) tokens
            # there is no need to save/upload it with adapter
            tk_source = config['sparta_pretrained_model'] # base model
        self.tokenizer = AutoTokenizer.from_pretrained(tk_source, padding_side='left')
        assert(len(self.tokenizer) == self.base_model.config.vocab_size)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.template = input_template 

    def __repr__(self):
        id2label = [f"{id}: '{label}'" for id, label in self.id2label.items()]
        return (f"(SpaRTA)ModelForSeqClassification(\n"
                f"\tadapter = '{self.adapter}'\n"
                f"\tmodel = '{self.base_model.config._name_or_path}'\n"
                f"\tid2label = {{{', '.join(id2label)}}}\n)")

    @torch.no_grad()
    def __call__(self, input_ids, attention_mask=None):

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # use last_hidden_state of last token in input sequence to do classification
        with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
            last_hidden_state = self.base_model(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                use_cache=False).last_hidden_state[:,-1,:]

            logits = self.head(last_hidden_state.to(self.head.weight.dtype))

        return logits.float()

    @torch.no_grad()
    def classify(self, texts): 
        if self.template:
            texts = [self.template.format(**text) for text in texts]        
        model_inputs = self.tokenizer(texts, return_tensors='pt', padding=True)
        logits = self(**model_inputs)
        class_probs = logits.softmax(-1)  
        return class_probs

    @torch.no_grad()
    def decide_class(self, texts): #  predicts classification labels
        class_probs = self.classify(texts)
        id_predictions = torch.argmax(class_probs, dim=-1).tolist() # max prob ids
        label_predictions =  [self.id2label[str(i)] for i in id_predictions] # ids into labels
        return label_predictions # most likely labels

    @torch.no_grad()
    def evaluate(self, texts, labels, batch_size=1):
        n = len(texts)
        labels = torch.tensor(labels).to(self.device)
        loss, predictions, accuracy = 0., [], 0.
        for i in range(0, n, batch_size): 
            batch_end = min(i + batch_size, n)
            batch_texts = texts[i:batch_end]
            batch_labels = labels[i:batch_end]
            class_probs = self.classify(batch_texts)
            loss += - class_probs[torch.arange(class_probs.shape[0]), batch_labels].log().sum().item()
            batch_predictions = torch.argmax(class_probs, dim=-1)
            accuracy += (batch_predictions == batch_labels).sum().item()
            predictions.extend(batch_predictions.tolist())
        # confusion matrix
        n_labels = self.head.out_features
        cm = torch.zeros((n_labels, n_labels), dtype=torch.int32)
        for label, pred in zip(labels.tolist(), predictions):
            cm[label, pred] += 1 
        results = {'loss': loss/n, 
                   'accuracy': accuracy/n, 
                   'confusion matrix': cm,
                   'balanced accuracy': bacc(cm)}
        if cm.shape == (2,2):
            results['MCC'] = mcc(cm)
            results['F1-score'] = f1_score(cm)
        return results
    
    def to(self, device):
        self.base_model.to(device)
        self.head.to(device)
        self.device = device



class SpaRTAforCausalLM:
    """Loads a Sparse Adapted Causal Model for Inference"""

    @torch.no_grad()
    def __init__(self, adapter, device_map='cpu'):

        self.adapter = adapter

        with open(os.path.join(adapter, 'config.json'), 'r') as f:
            config = json.load(f)

        base_model = AutoModelForCausalLM.from_pretrained(
                config['sparta_pretrained_model'],
                torch_dtype=torch.bfloat16,
                device_map=device_map).eval()

        # merge SpaRTA adapter into base model
        sparse_deltas = load_sparse_deltas(adapter)
        for name, param in base_model.named_parameters():
            indices = sparse_deltas['indices'][name].to(param.device).unbind(1)
            chosen_params = param[indices].float()
            deltas = sparse_deltas['deltas'][name].to(param.device).float()
            param[indices] = (chosen_params + deltas).to(param.dtype)
        del indices, deltas, sparse_deltas

        if not hasattr(base_model, 'device'):
            base_model.device = base_model.get_input_embeddings().weight.device

        self.base_model = base_model

    def __repr__(self):
        return (f"(SpaRTA)ModelForCausalLM(\n"
                f"\tadapter = '{self.adapter}'\n"
                f"\tbase model = '{self.base_model.config._name_or_path}'\n)")

    def __getattr__(self, name):
        return getattr(self.base_model, name)

    def __call__(self, *args, **kwargs):
        return self.base_model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        # model saved with adapter merged into base model
        self.base_model.save_pretrained(*args, **kwargs)


def load_sparse_deltas(adapter_dpath):
    st_fpath = os.path.join(adapter_dpath, 'sparse_deltas.safetensors')
    pt_fpath = os.path.join(adapter_dpath, 'sparse_deltas.pt')
    if os.path.isfile(st_fpath):
        data = safetensors.torch.load_file(st_fpath, device='cpu')
        sparse_deltas = {'indices': {}, 'deltas': {}}
        for k, v in data.items():
            if k.startswith('indices.'):
                sparse_deltas['indices'][k[8:]] = v
            elif k.startswith('deltas.'):
                sparse_deltas['deltas'][k[7:]] = v
            else:
                raise ValueError(f"Invalid key '{k}' in file '{st_fpath}'")
    elif os.path.isfile(pt_fpath):
        sparse_deltas = torch.load(pt_fpath, map_location='cpu', weights_only=True)
    else:
        raise FileNotFoundError(f"No SpaRTA adapter found in {adapter_dpath}")
    return sparse_deltas


def load_init_tensor(adapter_dpath, tensor_name, device):
    st_fpath = os.path.join(adapter_dpath, f'{tensor_name}_init.safetensors')
    pt_fpath = os.path.join(adapter_dpath, f'{tensor_name}_init.pt')
    if os.path.isfile(st_fpath):
        tensor = safetensors.torch.load_file(st_fpath, device=device)[tensor_name]
    elif os.path.isfile(pt_fpath):
        tensor = torch.load(pt_fpath, map_location=device, weights_only=True)
    else:
        raise FileNotFoundError(f"No init tensor for {tensor_name} found in {adapter_dpath}")
    return tensor



def convert_pt_to_safetensors(adapter_fpath):
    pt_fpath = os.path.join(adapter_fpath, 'sparse_deltas.pt')
    sparse_deltas = torch.load(pt_fpath, map_location='cpu', weights_only=True)

    data = {} # tensors
    for k, v in sparse_deltas['indices'].items():
        data[f"indices.{k}"] = v.contiguous()
    for k, v in sparse_deltas['deltas'].items():
        data[f"deltas.{k}"] = v.contiguous()

    st_fpath = os.path.join(adapter_fpath,'sparse_deltas.safetensors')
    safetensors.torch.save_file(data, st_fpath)
