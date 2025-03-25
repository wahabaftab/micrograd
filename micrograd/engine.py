import math 
import numpy as np

class Value:

  def __init__(self, data, children=(), op="",label=''):
    self.data = data
    self.grad= 0
    self.prev = (children)
    self.operator= op
    self.label=label
    self.H=0.000001
    self.parent = set()  # Track which parents contributed to grad
    self.grad_contributions = {}  # Track gradients per parent
  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data , (self , other), "+")

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data, (self, other), "*")

  def __rmul__(self,other):
    return self.__mul__(other)

  def __radd__(self,other):
    return self.__add__(other)

  def __pow__(self,other):
    assert isinstance(other, (int,float))
    return Value(self.data ** other, (self,), "**"+str(other))

  def __truediv__(self, other):
    return self * other**-1

  def __rtruediv__(self, other): # other / self
      return other * self**-1

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self,other):
    return self.__sub__(other)

  def tanh(self):
    x=self.data
    t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
    return Value(t, (self,), "tanh")

  def exp(self):
    return Value(math.exp(self.data), (self,), "exp")


  def relu(self):
      out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

      return out

  def backprop(self,prev_der=1, first_recursion=True):
    if first_recursion:
      self.grad=1
      prev_der=self.grad

    h=self.H

    if not self.prev:
      return
    pow_operator=''
    pow_value=0
    if '**' in self.operator:
      pow_operator = self.operator[:2]  # First two characters are '**'
      pow_value = int(self.operator[2:])  # The remaining part is the integer


    #if self has 2 child
    if len(self.prev)==2:
      child1,child2=self.prev

    #if this parent has already updated gradient before
      if self in child1.parent:
          # Subtract previous contribution from this parent before updating, ensures that same parent doesnt update gradient of child multiple times
          if self in child1.grad_contributions:
              child1.grad -= child1.grad_contributions[self]

    #if this parent has already updated gradient before
      if self in child2.parent:
          # Subtract previous contribution from this parent before updating, ensures that same parent doesnt update gradient of child multiple times
          if self in child2.grad_contributions:
              child2.grad -= child2.grad_contributions[self]

      child1.parent.add(self)
      child2.parent.add(self)

      temp1= child1.__add__(Value(h))
      temp2= child2.__add__(Value(h))



    #if self has 1 child
    elif len(self.prev)==1:
      child1,=self.prev
      if self in child1.parent:
          # Subtract previous contribution from this parent before updating, ensures that same parent doesnt update gradient of child multiple times
          if self in child1.grad_contributions:
              child1.grad -= child1.grad_contributions[self]
      child1.parent.add(self)

    if self.operator == "+":
      temp1=temp1.__add__(child2)
      child1_new=child1.__add__(child2)
      temp2=temp2.__add__(child1)
      child2_new=child2.__add__(child1)

    elif self.operator == "*":
      temp1=temp1.__mul__(child2)
      child1_new=child1.__mul__(child2)
      temp2=temp2.__mul__(child1)
      child2_new=child2.__mul__(child1)

    elif pow_operator == '**' :
      child1_der=pow_value * (child1.data**(pow_value-1)) * prev_der
      child1.grad= child1.grad+child1_der
      child1.grad_contributions[self] = child1_der  # Store new contribution
      child1.backprop(child1.grad,False)

    elif self.operator == "tanh":
      child1_der= 1-(self.data**2) * prev_der
      child1.grad= child1.grad+child1_der
      child1.grad_contributions[self] = child1_der  # Store new contribution
      child1.backprop(child1.grad,False)

    elif self.operator == "exp":

      child1_der= child1.exp().data * prev_der
      child1.grad= child1.grad+child1_der
      child1.grad_contributions[self] = child1_der  # Store new contribution
      child1.backprop(child1.grad,False)

    elif self.operator == "ReLU":

      child1_der= (self.data > 0) * prev_der
      child1.grad= child1.grad+child1_der
      child1.grad_contributions[self] = child1_der  # Store new contribution
      child1.backprop(child1.grad,False)

    if self.operator in  ('*','+'):
      child1_der= ((temp1.data  - child1_new.data) / h)*prev_der
      child2_der= ((temp2.data  - child2_new.data) / h)*prev_der
      child1.grad= child1.grad+child1_der
      child2.grad= child2.grad+child2_der
      child1.grad_contributions[self] = child1_der  # Store new contribution
      child2.grad_contributions[self] = child2_der  # Store new contribution
      child1.backprop(child1.grad,False)
      child2.backprop(child2.grad,False)
