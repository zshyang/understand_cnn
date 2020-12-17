# understand_cnn
Reproduce the paper "Visualizing and Understanding Convolutional Networks"

And the implementation is transformed from a `Kera` based repo.

https://github.com/saketd403/Visualizing-and-Understanding-Convolutional-neural-networks

By run the following command you should be able to get the figures.

```shell
python main.py
```

And here is an example image to visualize the 4th layer in the network.

![layer 4](/home/george/04.png)

## Issues:

1. The bias term of convolution is never used.
2. The linear layer is hard to be visualized.
3. Start from 16th layer, the visualization is becoming harder and harder.