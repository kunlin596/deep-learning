# Transformer

#date/2017 #publisher/Google

## Introduction

Paper: [[wiki#^c69f65]]

[[wiki#^18a29b|Dominant sequence transduction]].

### RNN

RNN is sequential, therefore it's hard to be parallelized.

$$
h(t)=f(h(t-1))
$$

Sequential information in the earlier time might be lost because the information is propagated sequentially.

### Attention Mechanism

It's used in encoder-decoder architecture for effectively transfer information from encoder to decoder.

### Transformer

Discard recurrence and reply entirely on [[wiki#^8bb295|attention]] mechanism. Convolutional layers have the advantage of have multiple channels which represent multiple learned patterns. Transformer achieve this effects by using [[wiki#^cf895e|multi-head attention]] for learning multiple patterns.

Encoder maps an input sequence of symbol representations $(x_1,x_2,...,x_n)$ to a sequence of continuous representations $\textbf{z}=(z_1,...,z_n)$. Here $x_i$ could be a word in a sentence, and each generated $z_i$ is a vector representation of that word.

Given $\textbf{z}$ , the decoder then generates an output sequence $(y_1,...,y_m)$ of symbols __one element__ at a time. At each step the model is [[wiki#^16bab0|auto-regressive]].

## Details

### Word Embedding

See [[wiki#^efe82a]].

### Encoder and Decoder Stacks

#### Encoder

First part is a [[wiki#^cf895e|multi-head self-attention mechanism]], the second part is a simple [[wiki#^f03b40|MLP]]. A [[resnet#^84e654|residual connection]] and then [[wiki#^0598d6|layer normalization]] is employed on the outputs.

#### Decoder

The [[wiki#^cf895e|masked multi-head attention mechanism]] ensures that at time $t$ we won't see the future inputs.

### Attention

See [[wiki#^4b4f84]]. Self-attention has fewer assumptions, and because of this, usually transformer models are usually larger than CNN/RNN.

### Point-wise Feed-Forward  Networks

$$
\mathrm{FFN}(x)=\mathrm{max}(0,xW_1+b_1)W_2+b_2
$$

### Positional Encoding

No spatial/sequential information in these weighted sum. So add $i$ into the result.
$$
\begin{align}
\mathrm{PE}_{(pos,2i)}&=sin(pos/10000^{2i/d_{model}}) \\
\mathrm{PE}_{(pos,2i+1)}&=cos(pos/10000^{2i/d_{model}}) \\
\end{align}
$$
### Optimizer
[[wiki|Adam]] is not sensitive to learning rate. In this paper, the learning rate is computed through an equation.

### Regularization

#### Label Smoothing

[[wiki|Label smoothing]] is introduced from [[Inception V3]].
