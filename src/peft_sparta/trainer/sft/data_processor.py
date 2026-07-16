import torch

class DataProcessor:
    
    def __init__(self, tokenizer, task):
        self.tokenizer = tokenizer
        self.task = task
        self.tokenizer.padding_side = 'left' if self.task == 'SEQ_CLS' else 'right' # CAUSAL_LM


    def build_dataloader(self, dataset, batch_size, shuffle=False, max_token_len=None): 

        if dataset is None:
            return None

        # tokenization + labeling + set up collator
        if self.task == 'SEQ_CLS':
            dataset, collate_fn = self.prepare_seq_cls(dataset)
        elif self.task == 'CAUSAL_LM':
            dataset, collate_fn = self.prepare_causal_lm(dataset)
        else:
            raise NotImplementedError(f"Task: {self.task} not implemented")

        if max_token_len:
            dataset = dataset.filter(
                lambda example: len(example['input_ids']) <= max_token_len,
                batched=False,
            )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            collate_fn = collate_fn
        )


    def prepare_seq_cls(self, dataset):

        assert 'text' in dataset.column_names and 'label' in dataset.column_names, \
            "SEQ_CLS dataset must have 'text' and 'label' columns"

        tokenize_fn = lambda example: {'input_ids': self.tokenizer.encode(example['text'])}
        dataset = dataset.map(tokenize_fn, batched=False)
        dataset = dataset.remove_columns(['text'])
        dataset = dataset.rename_column('label', 'labels')

        collate_fn = lambda examples: self.tokenizer.pad(examples, return_tensors='pt')
          
        return dataset, collate_fn


    def prepare_causal_lm(self, dataset):

        # dataset format detection
        if 'messages' in dataset.column_names:
            
            # prompt-completion using chat template format: for instruction-following
            def tokenize_prompt_completion_chat(example):  
                messages = example['messages']
                assert len(messages) >= 2, "prompt + response needed in every example"
                assert all('role' in m and 'content' in m for m in messages), "incorrect messages format"
                assert messages[-1]['role'] == 'assistant', "last message should be the assistant's"
                assert len(messages[-1]['content']) > 0, "assistant response is empty"
                # full 1-turn conversation: system + user + assistant header+content+eot
                input_ids = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False, return_dict=False,
                )
                # prompt only: system + user + assistant header: no assistant content
                prompt_ids = self.tokenizer.apply_chat_template(
                    messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False,
                )
                assert len(prompt_ids) < len(input_ids), "no assistant tokens to supervise"
                labels = list(input_ids)
                # supervise assistant last-response only (not the prompt):
                # loss to be computed over assistant last-turn (content) tokens only
                labels[: len(prompt_ids)] = [-100] * len(prompt_ids)
                return {"input_ids": input_ids, "labels": labels}
            
            tokenize_fn = tokenize_prompt_completion_chat 

        elif all(col in dataset.column_names for col in ['prompt','completion']):

            # prompt-completion (no chat)
            def tokenize_prompt_completion(example):
                assert len(example['prompt']) > 0, "prompt is empty"
                assert len(example['completion']) > 0, "completion is empty"
                prompt_ids = self.tokenizer.encode(example['prompt'])
                safe_method = True
                if safe_method: # slower processing
                    input_ids = self.tokenizer.encode(example['prompt'] + example['completion']) + [self.tokenizer.eos_token_id]
                    # loss to be computed only on completion tokens
                    labels = [-100] * len(prompt_ids) + list(input_ids[len(prompt_ids):])
                else:
                    completion_ids = self.tokenizer.encode(example['completion'], add_special_tokens=False)
                    input_ids = prompt_ids + completion_ids + [self.tokenizer.eos_token_id]
                    # loss to be computed only on completion tokens
                    labels = [-100] * len(prompt_ids) + completion_ids + [self.tokenizer.eos_token_id]
                assert len(prompt_ids) < len(input_ids), "no completion tokens to supervise"
                return {"input_ids": input_ids, "labels": labels}
            
            tokenize_fn = tokenize_prompt_completion

        elif 'text' in dataset.column_names:
        
            # language modeling (no chat): used for continued pretraining or domain adaptation 
            def tokenize_text(example): 
                assert len(example['text']) > 0, "text is empty"
                input_ids = self.tokenizer.encode(example['text'])
                labels = list(input_ids)  # supervise all tokens
                return {"input_ids": input_ids, "labels": labels}
            
            tokenize_fn = tokenize_text

        else:
            # supported data formats for training
            raise ValueError(
                "CAUSAL_LM dataset must have\n - a 'messages' column,\n "
                "- 'prompt'+'completion' columns, or\n - a 'text' column,\n"
                "representing our supported CAUSAL_LM data formats")

        dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

        def collate(batch):
            maxlen = max(len(x["input_ids"]) for x in batch)
            input_ids, labels, attn = [], [], []
            for x in batch:
                pad_len = maxlen - len(x["input_ids"])
                if self.tokenizer.padding_side == 'right': # for training and evaluation
                    input_ids.append(x["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
                    labels.append(x["labels"] + [-100] * pad_len)
                    attn.append([1] * len(x["input_ids"]) + [0] * pad_len)
                else: # left-padding for generation
                    input_ids.append([self.tokenizer.pad_token_id] * pad_len + x["input_ids"])
                    labels.append([-100] * pad_len + x["labels"])
                    attn.append([0] * pad_len + [1] * len(x["input_ids"]) )
            return {
                "input_ids":      torch.tensor(input_ids),
                "labels":         torch.tensor(labels),
                "attention_mask": torch.tensor(attn),
            }

        def collate_fast(batch):
            batch = [{k: torch.tensor(v) for k, v in x.items()} for x in batch]
            length_of_first = batch[0]['input_ids'].size(0)
            if all(x['input_ids'].size(0) == length_of_first for x in batch): # same length, no padding
                input_ids = torch.stack([x['input_ids'] for x in batch], dim=0)
                labels = torch.stack([x['labels'] for x in batch], dim=0)
            else: # padding
                max_length = max(x['input_ids'].size(0) for x in batch)
                input_ids = batch[0]['input_ids'].new_full([len(batch), max_length], self.tokenizer.pad_token_id)
                labels = batch[0]['labels'].new_full([len(batch), max_length], -100)
                for i, x in enumerate(batch):
                    if self.tokenizer.padding_side == 'right': # for training and evaluation
                        input_ids[i, : x['input_ids'].shape[0]] = x['input_ids']
                        labels[i, : x['labels'].shape[0]] = x['labels']
                    else: # left for generation
                        input_ids[i, -x['input_ids'].shape[0] :] = x['input_ids']
                        labels[i, -x['labels'].shape[0] :] = x['labels']
            return {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": (input_ids != self.tokenizer.pad_token_id).long()
                }

        collate_fn = collate_fast

        return dataset, collate_fn
