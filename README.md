# PEFT-SpaRTA

SpaRTA (Sparse Random parameTer Adaptation) is a Parameter-Efficient Fine-Tuning (PEFT) alternative to traditional LoRA that reduces the number of trainable parameters by randomly selecting a very small proportion of the model parameters to train on.

This Python package for now provides the invocation code necessary to load and run SpaRTA adapted models for inference. For an implementation of SpaRTA adapters and their training see https://github.com/IBM/sparta. For more details on how SpaRTA works see our [paper](https://arxiv.org/pdf/2502.15975). 


## Installation

```bash
pip install -i https://test.pypi.org/simple/ peft-sparta
```

## How to use it

### Download SpaRTA adapter from HF repository

Let's download a [SpaRTA adapter](https://huggingface.co/jesusriosal/sparta-gemma_2b-sst2) that spacializes the *google/gemma-2b* model to do *sentiment classification* of English sentences.

```bash

ADAPTER_DIR='/my_sparta_adapters/sparta-gemma_2b/'

mkdir -p $ADAPTER_DIR

hf download jesusriosal/sparta-gemma_2b-sst2 --local-dir $ADAPTER_DIR

```


### Load the SpaRTA adapter and create the adapted model

 ```python
from peft_sparta import SpaRTAforSequenceClassification

adapter_dir = '/my_sparta_adapters/sparta-gemma_2b/'

model = SpaRTAforSequenceClassification(
    adapter = adapter_dir,
    device = 'cuda')

print(model)
```

```none
(SpaRTA)ModelForSeqClassification(
	adapter = '/my_sparta_adapters/sparta-gemma_2b/'
	model = 'google/gemma-2b'
	id2label = {0: 'negative', 1: 'positive'}
)

```

### Inputs
Let's use our adapted model to classify a few sentences. For this adapter, the model consumes the sentences directly. No formating is needed  

```python

sentences = ["I enjoyed very much the movie.", 
             "It was painful to watch.", 
             "I couldn't enjoy more the movie.",
             "It was a bad movie."]
```

The model gives us the probabilities that each sentence (row) has negative (first column) or positive (second column) sentiment.

```python

class_probs = model.classify(sentences) 

print(class_probs)

```none
tensor([[0.1152, 0.8848],
        [0.9497, 0.0503],
        [0.1689, 0.8311],
        [0.9720, 0.0280]], device='cuda:0')
```

To identify which column correspond to each class, use:

```python
print(model.id2label)
```
```none
{'0': 'negative', '1': 'positive'}
```



```

model.decide_class(cases) # for the predicted classes of each case

```


 ## Citation

```bibtex
@article{rios2025sparsity,
  title={Sparsity may be all you need: Sparse random parameter adaptation},
  author={Rios, Jesus and Dognin, Pierre and Luss, Ronny and Ramamurthy, Karthikeyan N},
  journal={arXiv preprint arXiv:2502.15975},
  year={2025}
}

```




