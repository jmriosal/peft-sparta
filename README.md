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

model = SpaRTAforSequenceClassification(
    adapter = "/my_sparta_adapters/sparta-gemma_2b/", 
    device = 'cuda',
)

cases = ["list of input cases",
          "in the format used during training",
          "consumed by the input_template"]

model.classify(cases) # for class probs of each case

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




