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
### Inference 

#### Classify (probabilistically)

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

Here is probability that each sentence has a positive sentiment given by the model  

```python
for sentence, pos_prob in zip(sentences, class_probs[:,1]):
    print(f"{pos_prob.item()*100:>4.0f}%\t{sentence}")
```

```none
 Prob   Sentence
 ----   -----------------------------
  88%	I enjoyed very much the movie.
   5%	It was painful to watch.
  83%	I couldn't enjoy more the movie.
   3%	It was a bad movie.
```



#### Decide on classes

We have seen how the model makes probabilistic assessments of the sentiment of each sentence. If we want the model to make a decison on whether the sentence has positive or negative sentiment, we can use: 

```python
classes = model.decide_class(sentences) 
```

to obtaine the model's predicted class of each sentence. Basically, the model takes the most likely class as its sentiment prediction of a sentence
```python
for sentence, sentence_class in zip(sentences, classes):
    print(f"'{sentence_class}':  {sentence}")
```

```none
 Sentiment   Sentence
-----------  -------------------------------
'positive':  I enjoyed very much the movie.
'negative':  It was painful to watch.
'positive':  I couldn't enjoy more the movie.
'negative':  It was a bad movie.
```

### Input templates

Sometimes the input to the model may need to be formated before our adapted model can processs it. 
This is typicaly the case when using instruction-following models, for which wrapping the input within an instruction, formated with the model chat template, can be advantageous. In these cases, we can use the following *input_template* argument to format raw inputs into the format used during training. 

To see this, let's use [another SpaRTA adapter](https://huggingface.co/jesusriosal/sparta-gemma_2b_it-sst2) for *sentiment classification* based on the [google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it) model.  


```bash
hf download jesusriosal/sparta-gemma_2b-sst2 --local-dir '/my_sparta_adapters/sparta-gemma_2b_it/'
```

```python

from peft_sparta import SpaRTAforSequenceClassification

adapter_dir = '/my_sparta_adapters/sparta-gemma_2b_it/'

model = SpaRTAforSequenceClassification(
    adapter=adapter_dir,
    device='cuda',
    input_template = ("<start_of_turn>user\n"
                      "Determine the sentiment of the following sentence about a movie. "
                      "The sentiment can only be classified as positive or negative.\n"
                      "Sentence: {sentence}" 
                      "<end_of_turn>\n<start_of_turn>model\n"
                      "The sentiment of the sentence is")
    )

print(model)
```

```none
(SpaRTA)ModelForSeqClassification(
	adapter = '/my_sparta_adapters/sparta-gemma_2b_it/'
	model = 'google/gemma-2b-it'
	id2label = {0: 'negative', 1: 'positive'}
)
```

This SpaRTA adapter was trained formating the input sentences to be classified with the *input_template* (see *model.template* below), which included the task instruction. This ensures that during inference the same formatting is used on the inputs to be classified.


```python
print(model.template)
```

```none
<start_of_turn>user
Determine the sentiment of the following sentence about a movie. The sentiment can only be classified as positive or negative.
Sentence: {sentence}<end_of_turn>
<start_of_turn>model
The sentiment of the sentence is
```


For example, the sentence 

```none
I enjoyed very much the movie.
```

is converted to 

```none
<start_of_turn>user
Determine the sentiment of the following sentence about a movie. The sentiment can only be classified as positive or negative.
Sentence: I enjoyed very much the movie.<end_of_turn>
<start_of_turn>model
The sentiment of the sentence is
```
before passing it to the model for classification 


Thus, to classify the (raw, non-formatted) sentences [above](#inputs)

 


## Citation

```bibtex
@article{rios2025sparsity,
  title={Sparsity may be all you need: Sparse random parameter adaptation},
  author={Rios, Jesus and Dognin, Pierre and Luss, Ronny and Ramamurthy, Karthikeyan N},
  journal={arXiv preprint arXiv:2502.15975},
  year={2025}
}

```




