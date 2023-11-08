# Transformers

## Introduction

This repository is a summary of sorts, as well as a collection of notes and experiments, a snapshot of my current stage of understanding transformer models, and a glimpse of how I like to structure my modeling code. First and foremost, this code is based on Kaparthy's nanoGPT repo (https://github.com/karpathy/nanoGPT) and the accompanying YouTube walk-through (link), which I found to be an excellent starting point - no-frills PyTorch model code and multi-GPU configuration, and some gems sprinkled here and there such as Flash attention, weight-tying, and ready to use GPT-2 loading.

Nevertheless, this was only a starting point. I split the monolithic script into dataset, tokenizer, model, trainer sections, and expanded the config-based runner to allow for a well-documented experiment setup. Modularizing this code makes it easier to exchange the GPT model with a BERT or RoBERTa model, or switch out the character-level tokenizer with a byte-pair encoding setup, or some other fancy pre-trained module. The PyTorch DataLoader seems like a useful feature for maximizing compute utilization in larger training runs, and inheriting a base class means one can easily configure different dataset classes for different datasets. Theoretically, this means the jump from text to image to multi-modal datasets shouldn't be too crazy.

Recently, PyTorch released a matrix multiplication visualizer, which I sought to take advantage of here to provide a visual step-by-step approach to the attention mechanism. Screenshots for a specific attention implementation can be found in the attention notebook. The other notebooks are my way of testing and explaining what else is going on in the code. Sometimes, this is just a matter of checking the dimensions of the tensor going  in and out of a module,

## Contents

### Modules

#### Datasets

The dataset base class templates the necessary methods. A PyTorch dataset requires the `len` and `getitem` methods, and the `load_dataset` method can be utilized to import the dataset from file. Additionally, a `download_dataset` can be helpful when running the dataloader for the first time.

```
class Dataset(ABC):
    """Dataset base class to be inherited by all dataset implementations."""

    def __init__(self, data_dpath: str) -> None:
        self.data_dpath = data_dpath
        self.data = []

    @abstractmethod
    def _load_dataset(self):
        """Load dataset from file."""

    @abstractmethod
    def __len__(self):
        """Return dataset length."""

    @abstractmethod
    def __getitem__(self, idx):
        """Index individual dataset item."""
```

The idea here is that all specific dataset implementations such as **Shakespeare** or **IMDB** can inherit from this class but provide unique loading capabilities.

#### Tokenizers

Same idea here, we use a base class to template the tokenizer's functionality - it needs to load the token mapping, return the vocab size, encode a text input and decode a list of tokens.

```
class Tokenizer(ABC):
    """Tokenizer base class to be inherited by all tokenizers."""

    def __init__(self, tokenizer_dpath: str) -> None:
        self.tokenizer_dpath: str = tokenizer_dpath
        self.vocab_size: int = 0

    def __len__(self):
        """Return vocab size."""
        return self.vocab_size

    @abstractmethod
    def _load_tokenizer(self):
        """Load tokenizer from file"""

    @abstractmethod
    def _build_tokenizer(self):
        """Build tokenizer from scratch"""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Return token to id mapping for text input."""

    @abstractmethod
    def decode(self, vec: List[int]) -> str:
        """Return id to token mapping for array of ids."""

```


#### Models

Again, we have a base class, but this time it is not our own implementation - we can just rely on PyTorch.

```
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

The `nn.Module` provides a ready to train and optimize neural network module that can be nested into a tree structure. This makes writing custom modules super easy, as shown by Kaparthy's implementation. But switching from autoregressive self-attention to regular self-attention should be easy enough.

Currently, we have the following modules implemented within the models directory
* `GPT`: Generative Pre-trained Transformer utilizing Kaparthy's GPT-2 implementation with a language modeling head that can do next token prediction.
* `Attention`: Containsa CausalSelfAttention module that can be called sequentially in blocks.
* `Utils`: General utility functions that can be used across transformer modules.


### Configs

Whem we run an experiment, we want to be able to log the configuration to the experiment directory, so experiments across different architectures can be compared and repeated. Loading and logging the configurations is best handled in the trainer itself.


### Trainer

The trainer is what pieces the above components together. It is called to handle the training and validation loop. Currently, it is also used to host the generative methods, but this will be adjusted soon, with the addition of a predictor class.


### Runner

We need a central script to load the action (train, sample, predict) and instantiate the corresponding handler class. This is best done using the handy jsonargparse library in a runner script, that can be called from an executable file. I've seen more involved utilizations of the jsonargparse libary for class instantiation, however, these were also kind of messy. For now, I'm liking the simplicity of this approach.

## Notebooks

### Attention

We utilize the following notation

Given input $X \in R[B, T, C]$ we compute the attention 

```math
\text{Attention(Q, K, V)} = \text{softmax}({\frac{QK^T}{\sqrt{\text{n emb}}}}) V
```

where

g```math
Q = XW_q + b_q
K = XW_k + b_k
V = XW_v + b_v
```

Thanks to the matrix multiplication visualizer, we can take a specific parameterization, such as $[B, T, C] = [1, 5, 2]$ to visualize the mechanism end to end.

![Attention](images/end_to_end_attention.png)

More details are provided in a step-by-step walkthrough in the notebook including
* General Single Head Attention
* GPT-Style (autoregressive) Single Head Attention
* Scaled Single Head Attention
* Scaled Multi-Head Attention
* Output Projection


### GPT

Walkthrough of the GPT module, including features such as
* Layer normalization
* GELU
* MLP
* Causal Self-Attention
* Transformer Model Dict
* Language Modeling Head
* Configure Optimizer
* Model Flop Utilization
* Generate Tokens using next token prediction

### Model Tests

Checks that model performs training and inference step as expected.

### Dataset Tests

Check dataloader and get item method.

