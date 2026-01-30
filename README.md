# PEFT-SpaRTA

SpaRTA (Sparse Random parameTer Adaptation) is a Parameter-Efficient Fine-Tuning (PEFT) alternative to traditional LoRA that reduces the number of trainable parameters by randomly selecting a very small proportion of the model parameters to train on.

This Python package provides the invocation code necessary to load and run SpaRTA adapted models for inference. For an implementation of SpaRTA adapters and their training see https://github.com/IBM/sparta. For more details on how SpaRTA works see our [paper](https://arxiv.org/pdf/2502.15975). 


## Installation

```bash
pip install -i https://test.pypi.org/simple/ peft-sparta
```

## How to use it

### Download a SpaRTA adapter from a Hugging Face repository

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

#### Probabilistic classification

The adapted model can give us its estimated probabilities that each sentence (row) has negative (first column) or positive (second column) sentiment.

```python

class_probs = model.classify(sentences) 

print(class_probs)
```

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

Here are the model's estimated probabilities of positive sentiment for each sentence

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



#### Deciding the sentiment class of each sentence (deterministic classification)

We have seen how the model makes probabilistic assessments of the sentiment of each sentence. If we want the model to make a definitive decison on whether the sentence has positive or negative sentiment, we can use: 

```python
classes = model.decide_class(sentences) 
```

to obtain the model's predicted class of each sentence. Basically, the model takes the most likely class as its sentiment prediction of a sentence
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

Sometimes the input to the model may need to be formatted before our adapted model can processs it. 
This is typicaly the case when using instruction-following models, for which wrapping the input within an instruction, formatted with the model's chat template, can be advantageous. In these cases, we can use the following *input_template* argument to specify the formatting over raw inputs used during training, and needed during inference. 

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

This SpaRTA adapter was trained formating the input sentences to be classified with the *input_template* (see *model.template* below), which included a task instruction. This ensures that during inference the same formatting is used on the inputs to be classified.


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


Thus, to classify the (raw, non-formatted) sentences [above](#inputs) we proceed as follows

```python
sentences = [{'sentence': sent} for sent in sentences]

class_probs = model.classify(sentences)

# prob of positive sentiment for each sentence 
for sentence, pos_prob in zip(sentences, class_probs[:,1]):
    print(f"{pos_prob.item()*100:>4.0f}%\t{sentence['sentence']}")
```
 
```none
 100%	I enjoyed very much the movie.
   0%	It was painful to watch.
 100%	I couldn't enjoy more the movie.
   0%	It was a bad movie.
```

```python
classes = model.decide_class(sentences)

for sentence, sentence_class in zip(sentences, classes):
    print(f"'{sentence_class}':  {sentence['sentence']}")

```

```none
 Sentiment   Sentence
-----------  -------------------------------
'positive':  I enjoyed very much the movie.
'negative':  It was painful to watch.
'positive':  I couldn't enjoy more the movie.
'negative':  It was a bad movie.
```



### Out-of-Distribution performance evaluations 

If you have a labeled dataset with English sentences and their sentiment labels, like the one below, you can evaluate the performace of these models on that dataset as follows. 

Given the following dataset of new, unseen sentences and their sentiment labels:

```python
test_sentences = ["it's a charming journey. ",
                  "bleak and desperate",
                  "nolan is poised to embark a major career as a commercial yet inventive filmmaker.",
                  "the acting, costumes, music, cinematography and sound are all astounding. ",
                  "it's slow -- very, very slow. ",
                  "the film is a refreshingly serious look at young women.",
                  "a sometimes tedious film.",
                  "like doing last year's taxes with your ex-wife.",
                  "you don't have to know about music to appreciate the film. ",
                  "in exactly 89 minutes, most of which passed as slowly as if i'd been sitting naked on an igloo, the movie sank from quirky to jerky to utter turkey."]

test_labels = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]

```
where a label of 0 represents *negative* sentiment and a label of 1 *positive*.

We evaluate the performance of the model on this labeled dataset as follows, after putting each sentence within a dictionary with a key named 'sentence' to be consumed by the *input_template*.

```pyhton
test_sentences = [{'sentence': sent} for sent in test_sentences] # for the model with input_template

model.evaluate(test_sentences, test_labels, batch_size=64)
```
```none
loss: 0.002
accuracy: 100%
confusion matrix: [5, 0
                   0, 5]
balanced accuracy: 100% 
MCC: 1.0
F1-score: 1.0
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




