# Wikipedia

**Dominant sequence transduction** ^18a29b

Generate one string from another string, such as natural language translation.

**Auto-regression** ^16bab0

The previous outputs will be the current inputs.

**Batch Normalization**

#TODO

**Layer normalization** ^0598d6

#TODO

**Word Embedding** ^efe82a

[PyTorch Word Embedding](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)

print(hello_embed)
```

**Attention** ^8bb295

An attention can be described as mapping a query and a set of key-value pairs to an output where the query, keys, values and output are all vectors.

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

$QK^T$ is called _Scaled Dot-Product Attention_ which is the simplest attention mechanism. It's imply computing the "similarity" of the query and the key. In [[#|^c69f65]] $d_k$ is set to be 512. Suppose $Q$ is of shape $n\times d_k$ and $K$ is of shape $m \times d_k$, then $QK^T$ will be $n\times m$. $V$ is $m\times d_v$. The overall output from attention function will be $n\times d_v$.

The two most commonly used attention functions additive attention and dot-product (multiplicative) attention (the algorithm described above).

$\sqrt{d_k}$ is used when the vector is long (the values could be either very large or small). When softmax is applied, the value is pushed towards the either 0 or 1. In this case, the gradients will be small and it will create some trouble for optimization.

Mask here is to replace the value before entering the softmax function with a large negative number such as $1e^{-10}$. When this value enters softmax, it will become zero.

When the same values being used as both $Q,K,V$, it's called **self-attention**.

**Multi-head attention** ^cf895e

Given $h$ chances, we hope that we can learn $h$ patterns.

$$
\begin{align}
\mathrm{MultiHead}(Q,K,V)&=\mathrm{Concat}(\mathrm{head_1},...,\mathrm{head_h})W^O \\
where\space\mathrm{head_i}&=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)
\end{align}
$$

Where the projections are parameter matrices $W_i^Q\in \mathbb{R}^{d_{model}\times d_k},W_i^K\in\mathbb{R}^{d_{model}\times d_k}, W_i^V\in \mathbb{R}^{d_{model}\times d_v}$ and $W_O\in\mathbb{R}^{hd_v\times d_{model}})$. Here all $W_i$ matrices are learnable parameters.

In the original paper [[#^c69f65]], $h=8$. $d_k=d_v=d_{model}/h=64$. The final concatenated vector is $512$.

**Multi-layer perceptron (MLP)** ^f03b40

#TODO

## References

1 - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) ^c69f65
