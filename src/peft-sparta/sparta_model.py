import os
import json
import torch
from transformers import AutoModel, AutoTokenizer
from .metrics import mcc, f1_score, bacc

class SpaRTAforSequenceClassification:
    """Loads a Sparse Adapted Model for Inference"""

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
            assert(model.config._name_or_path == config['_name_or_path'])
            base_model = model.base_model
        else:
            # load from disk
            base_model = AutoModel.from_pretrained(config['_name_or_path'], 
                                                   torch_dtype=torch.bfloat16)
        base_model.eval().to(device)
        hidden_size = base_model.config.hidden_size

        # add new token embeddings if added to model
        new_embeddings_fpath = os.path.join(adapter, 'new_embeddings_init.pt')  
        if os.path.exists(new_embeddings_fpath):
            embeddings = base_model.get_input_embeddings()
            new_embeddings_init = torch.load(new_embeddings_fpath,
                                             map_location=embeddings.weight.device
                                             ).to(dtype=embeddings.weight.dtype)
            num_added_tokens = new_embeddings_init.shape[0]            
            with torch.no_grad():
                embeddings.weight.data = torch.cat((embeddings.weight.data, new_embeddings_init))
            embeddings.num_embeddings += num_added_tokens
            base_model.config.vocab_size += num_added_tokens
        
        # merge SpaRTA adapter into base model
        sparse_deltas = torch.load(os.path.join(adapter, 'sparse_deltas.pt'), map_location='cpu')
        for name, param in base_model.named_parameters():  
            name = base_model.base_model_prefix + '.' + name
            indices = sparse_deltas['indices'][name].to(device).unbind(1)
            # param[indices] += sparse_deltas['deltas'][name].to(device)
            chosen_params = param[indices].float()
            deltas = sparse_deltas['deltas'][name].to(device).float()
            param[indices] = (chosen_params + deltas).to(param.dtype)
        del indices

        # create sequence classification head
        w0 = torch.load(os.path.join(adapter,'head_init.pt'), map_location=device)
        num_labels = w0.shape[0]
        assert(num_labels == len(self.id2label))
        deltas = sparse_deltas['deltas']['score.weight'].detach().to(device)
        assert(len(deltas) // hidden_size == num_labels) # head fully trainable
        deltas = deltas.view(num_labels, hidden_size) 
        head = torch.nn.Linear(hidden_size, num_labels, bias=False).to(device)
        head.weight.data = (w0.float() + deltas.float()).to(dtype=head.weight.dtype) # w0 + deltas
        del w0, deltas, sparse_deltas
        
        self.base_model, self.head, self.device = base_model, head, device

        self.tokenizer = AutoTokenizer.from_pretrained(adapter, padding_side='left')         
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
        
