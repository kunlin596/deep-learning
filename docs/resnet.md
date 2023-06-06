# ResNet

[Video](https://www.youtube.com/watch?v=pWMnzCX4cwQ&list=PLFXJ6jwg0qW-7UM8iUTj3qKqdhbQULP5I&index=5) by Li Mu.

Training a deep network is hard, time and training performance. But deep networks are good.

Is learning better networks as easy as stacking more layers:

    - notorious problem of vanishing/exploding gradients, which hamper convergence from the beginning
    - this problem has been addressed by normalized initialization and intermediate normalization layers

## Degradation problem

With network depth increasing, accuracy gets saturated then degrades rapidly.

    - Not caused by over-fitting (both training and testing errors are higher when networks become complex)

If a shallow model works well, then adding more layers should not make the model worse -- because the added layers are _identity mapping_.

    - In reality, SGD cannot find such solution -- the added layers are all identity which means the deeper network is not degraded

### Construct such identity mapping explicitly

"Shortcut connections"
$$H(x)=F(x)+x$$

## Residual

In low-level CG/CV, PDEs, gradient boosting (learning through residuals, 20 years ago)

## Shortcut Connections

Originally from `Highway networks`.

## Residual Network

1x1 convolution -- convert space into channels, projection (netter performance)

### 10-crop testing (?)

## Details

In the loss graph, the thin red lines are the training loss. Initially it's higher than the validation error (thick plot), that's because it's we use a lot of data augmentation techniques, whereas during validation, there is no augmentation.

When the error plateaus, we decay the learning rate by 0.1 (empirically by human, require experience to decide when and where to decay, superstitious), that's the where the rapid loos drop comes from. When using the residual architecture, deeper network will get faster convergence and performance.

## Bottleneck
