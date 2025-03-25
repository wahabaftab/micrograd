
# About This Repository
This repository implements the same functionality as Andrej Karpathy's Micrograd, but with a different approach in the underlying code. Some key differences include:

Backpropagation using recursion: Instead of using topological sorting, I implemented backprop using recursion. This approach felt more intuitive to me while coding, even though it might be slightly less efficient. It still provides a good and different way to implement backpropagation.

Gradient computation for + and * operations: Instead of the approach used by Andrej, I compute gradients using the base derivative formula by adding a small increment h and applying the fundamental definition of a derivative:

𝐿
=
lim
⁡
ℎ
→
0
𝑓
(
𝑎
+
ℎ
)
−
𝑓
(
𝑎
)
ℎ
L= 
h→0
lim
​
  
h
f(a+h)−f(a)
​
 
This offers an alternative perspective on automatic differentiation while still producing correct results.

This project aims to showcase different ways to implement micrograd while maintaining clarity and correctness.




# micrograd

![awww](puppy.jpg)

A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

### Installation

```bash
pip install git+https://github.com/wahabaftab/micrograd.git
```
Notebook `install_demo.ipynb` contains sample code showcasing different functionalities.

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backprop()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

### Training a neural net

The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `micrograd.nn` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neuron](moon_mlp.png)

### Tracing / visualization

For added convenience, the notebook `trace_graph.ipynb` produces graphviz visualizations. I have added the code in **`graph.py`** which can be imported e.g. this one below is of a simple 2D neuron, and it shows both the data (top number in each node) and the gradient (bottom number in each node).

```python
from micrograd.engine import Value
from micrograd import nn
from micrograd.graph import visualize_graph
import random

random.seed(1337)

n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
y.backprop()
graph = visualize_graph(y)
display(graph)
```

![2d neuron](gout.svg)

### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python -m pytest
```

### License

MIT
