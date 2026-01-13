# PEFT-SpaRTA

SpaRTA (Sparse Random parameTer Adaptation) is a Parameter-Efficient Fine-Tuning (PEFT) alternative to traditional LoRA that reduces the number of trainable parameters by randomly selecting a very small proportion of the model parameters to train on.

This Python package provides for now the invocation code necessary to load and run SpaRTA adapted models. For an implementation of SpaRTA adapters and their training see https://github.com/IBM/sparta. For more details on how SpaRTA works see our [paper](https://arxiv.org/pdf/2502.15975). 


## Installation

```bash
pip install -i https://test.pypi.org/simple/ peft-sparta
```

## How to use it

 ```python
from peft_sparta import SpaRTAforSequenceClassification

model = SpaRTAforSequenceClassification(
    adapter = "adapter_fpath", 
    device = 'cuda', # cpu if cuda is not available
    input_template = "used to train the adapter, possible including a instruction", 
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




