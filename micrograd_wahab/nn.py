import random
from micrograd_wahab.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
  def __init__(self,nin, nonlin=True):
    self.w=[Value(random.uniform(-1,1)) for _ in range(nin)]
    # self.b=Value(random.uniform(-1,1))
    self.b = Value(0) ##only to replicate demo.ipynb code on github
    self.nonlin = nonlin

  def __call__(self,x):
    result=sum((i*w for i,w in zip(x,self.w)),self.b)
    # return result
    return result.relu() if self.nonlin else result ##only to replicate demo.ipynb code on github

  def parameters(self):
    return self.w+[self.b]

class Layer(Module):

  # def __init__(self, nin, nout):
  #   self.layer=[Neuron(nin) for i in range(nout)]
  def __init__(self, nin, nout, **kwargs):   ##only to replicate demo.ipynb code on github
      self.layer = [Neuron(nin, **kwargs) for _ in range(nout)]

  def __call__(self,x):
    out= [neuron(x) for neuron in self.layer]
    return out[0] if len(out) == 1 else out

  def parameters(self):
    return [p for layer in self.layer for p in layer.parameters()]


class MLP(Module):
  def __init__(self, nin , nout):
    sz=[nin]+nout
    # self.layers=[Layer(sz[i],sz[i+1]) for i in range(len(nout))]
    self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nout)-1) for i in range(len(nout))] ##only to replicate demo.ipynb code on github

  def __call__(self, x):
    for layer in self.layers:
      x= layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

