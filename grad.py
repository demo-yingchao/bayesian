import numpy as np


def logsumexp(val):
  m = np.max(val, -1, keepdims=True)
  ret = m + np.log(np.sum(np.exp(val-m), -1, keepdims=True))
  return ret

def _softmax(val):
  m = np.max(val, -1, keepdims=True)
  ret = np.exp(val-m)
  ret = ret/np.sum(ret, -1, keepdims=True)
  return ret

def shape_back(val, grad):
  '''for numpy broadcast mechanic, grad should turn back to original shape'''
  val_dim = len(np.shape(val))
  grad_dim = len(np.shape(grad))
  if val_dim != grad_dim:
    reduce_dim = tuple(range(grad_dim-val_dim))
    grad = np.sum(grad, axis=reduce_dim)

  val_shape = np.shape(val)
  grad_shape = np.shape(grad)
  if len(val_shape) == len(grad_shape) and val_shape != grad_shape:
    reduce_axes = []
    for i, (vs, gs) in enumerate(zip(val_shape, grad_shape)):
      if vs != gs:
        if vs == 1:
          reduce_axes.append(i)
        else:
          raise Exception('val shape {} is not consistent with grad shape {}'.format(val_shape, grad_shape))
    grad = np.sum(grad, axis=tuple(reduce_axes), keepdims=True)
  return grad
      

class Oprator:
 
  def __init__(self):
    pass

  @classmethod
  def run(self, inp_val, param=None):
    pass

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    pass


class NegOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, = inp_val
    return -x_val

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    grad = prev_grad
    return (-prev_grad, )

class AbsOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, = inp_val
    return np.abs(x_val)

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    grad = prev_grad
    x_val, = prev_val
    grad = np.sign(x_val)*grad
    return (grad, )

class AddOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, y_val = inp_val
    return x_val + y_val

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    grad = prev_grad
    x_val, y_val = prev_val
    return (grad, grad)

class SubOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, y_val = inp_val
    return x_val - y_val

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    grad = prev_grad
    x_val, y_val = prev_val
    return (grad, -grad)

class MulOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, y_val = inp_val
    return x_val * y_val

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    x_val, y_val = prev_val
    x_grad = y_val * prev_grad
    y_grad = x_val * prev_grad
    return (x_grad, y_grad)

class DivOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, y_val = inp_val
    return x_val / y_val

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    x_val, y_val = prev_val
    x_grad = 1/y_val * prev_grad
    y_grad = -x_val/(y_val*y_val) * prev_grad
    return (x_grad, y_grad)


class PowOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, y_val = inp_val
    return x_val ** y_val

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    x_val, y_val = prev_val
    x_grad = y_val * (x_val)**(y_val-1) * prev_grad
    y_grad = 0
    return (x_grad, y_grad)


class DotOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, y_val = inp_val
    return x_val @ y_val

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    x_val, y_val = prev_val
    if isinstance(prev_grad, np.ndarray) and prev_grad.ndim > 0:
      if y_val.ndim > 1:
        x_grad = prev_grad @ y_val.swapaxes(-1,-2)
      else:
        x_grad = prev_grad[:, None] * y_val
      if x_val.ndim > 1:
        y_grad = x_val.swapaxes(-1,-2) @ prev_grad
      else:
        y_grad = prev_grad * x_val[:, None]
    else:
      x_grad = prev_grad * y_val
      y_grad = prev_grad * x_val
    return (x_grad, y_grad)

class ExpOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, = inp_val
    return np.exp(x_val)

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    x_val, = prev_val
    x_grad = np.exp(x_val) * prev_grad
    return (x_grad, )


class LogOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, = inp_val
    return np.log(x_val)

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    x_val, = prev_val
    x_grad = prev_grad / x_val
    return (x_grad, )


class TransposeOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, = inp_val
    if param:
      axes = param['axes']
      return np.transpose(x_val, axes)
    else:
      if x_val.ndim > 1:
        return x_val.swapaxes(-2,-1)
      else:
        return x_val

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    x_val, = prev_val
    if param:
      orig_axes = param['axes']
      axes = [0]*len(orig_axes)
      for i, k in enumerate(orig_axes):
        axes[k] = i
      x_grad = np.transpose(prev_grad, axes)
    else:
      if prev_grad.ndim > 1:
        x_grad = prev_grad.swapaxes(-2,-1)
      else:
        x_grad = prev_grad
    return (x_grad, )

class InverseOp(Oprator):
  
  @classmethod
  def run(self, inp_val, param=None):
    x_val, = inp_val
    return np.linalg.inv(x_val)

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    x_val, = prev_val
#x_grad = prev_grad*(-np.linalg.inv(x_val)@np.linalg.inv(x_val))
    x_inv_T = np.linalg.inv(x_val).T
    x_grad = -x_inv_T@prev_grad@x_inv_T
    return (x_grad, )


class DetOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, = inp_val
    return np.linalg.det(x_val)

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    x_val, = prev_val
    prev_grad = np.sum(prev_grad)
    x_grad = np.sum(prev_grad)*np.linalg.det(x_val)*np.linalg.inv(x_val).T
    return (x_grad, )

class TraceOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, = inp_val
    return np.trace(x_val, axis1=-2, axis2=-1)

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    x_val, = prev_val
    I = np.eye(x_val.shape[-1])
    shape = list(np.shape(x_val))
    shape[-1]=shape[-2]=1
    I = np.tile(I, shape)
    x_grad = I*np.reshape(prev_grad, shape)
    return (x_grad,)


class TraceMeanOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, = inp_val
    return np.trace(x_val, axis1=-2, axis2=-1)/x_val.shape[-1]

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    x_val, = prev_val
    I = np.eye(x_val.shape[-1])/x_val.shape[-1]
    shape = list(np.shape(x_val))
    shape[-1]=shape[-2]=1
    I = np.tile(I, shape)
    x_grad = I*np.reshape(prev_grad, shape)
    return (x_grad,)


class TraceSumOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, = inp_val
    return np.trace(x_val)

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    x_val, = prev_val
    I = np.eye(x_val.shape[0])
    x_grad = I*prev_grad
    return (x_grad,)


class SumOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, = inp_val
    return np.sum(x_val)

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    x_val, = prev_val
    ones = np.ones(x_val.shape)
    x_grad = ones*prev_grad
    return (x_grad,)


class MeanOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, = inp_val
    return np.mean(inp_val)

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    x_val, = prev_val
    mean_ones = np.ones(x_val.shape)/np.size(x_val)
    x_grad = mean_ones*prev_grad
    return (x_grad,)

class ReluOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, = inp_val
    return x_val*(x_val>0)

  @classmethod
  def compute_grad(self, prev_grad=1, prev_val=None, param=None):
    x_val, = prev_val
    x_grad = prev_grad * (x_val > 0)
    return (x_grad, )

class LogSoftmaxOp(Oprator):

  @classmethod
  def run(self, inp_val, param=None):
    x_val, = inp_val
    denorm = logsumexp(x_val)
    return x_val - denorm

  @classmethod
  def compute_grad(self, labels=1, prev_val=None, param=None):
    x_val, = prev_val
    '''
    denorm = logsumexp(x_val)
    logprob = x_val - denorm
    grad = np.exp(logprob) - labels
    '''
    grad = _softmax(x_val) - labels
    return (grad, )

  @classmethod
  def compute_loss(self, op_val, labels):
    return -np.sum(op_val*labels)

class Node:
  def __init__(self, inp_var=None, op=None, param=None, name='node'):
    self.inp_var = inp_var
    self.op = op
    self.param = param
    self.name = name
    self.op_val = None
    self.tmp_val = None
    return

  def forward(self, inp_dict={}):
    inp_val = [v.forward(inp_dict) if isinstance(v, Node) else v for v in self.inp_var]
    self.tmp_val = tuple(inp_val)
    self.op_val = self.op.run(self.tmp_val, self.param)
    return self.op_val

  def update(self, prev_grad=1):
    prev_grad = shape_back(self.op_val, prev_grad)
    grad = self.op.compute_grad(prev_grad, self.tmp_val, self.param)
    self.grad = grad
    for (v, g) in zip(self.inp_var, grad):
      if isinstance(v, Node):
        v.update(g)
    return
    
  def __neg__(self,):
    inp_var = (self,)
    op = NegOp
    return Node(inp_var, op, name='neg_op')

  def __add__(self, other):
    inp_var = (self, other)
    op = AddOp
    return Node(inp_var, op, name='add_op')

  def __radd__(self, other):
    return self + other

  def __sub__(self, other):
    inp_var = (self, other)
    op = SubOp
    return Node(inp_var, op, name='sub_op')

  def __rsub__(self, other):
    return self - other

  def __mul__(self, other):
    inp_var = (self, other)
    op = MulOp
    return Node(inp_var, op, name='mul_op')

  def __rmul__(self, other):
    return self * other

  def __truediv__(self, other):
    inp_var = (self, other)
    op = DivOp
    return Node(inp_var, op, name='div_op')

  def __rtruediv__(self, other):
    inp_var = (other, self)
    op = DivOp
    return Node(inp_var, op, name='div_op')

  def __pow__(self, other):
    inp_var = (self, other)
    op = PowOp
    return Node(inp_var, op, name='pow_op')

  def __matmul__(self, other):
    inp_var = (self, other)
    op = DotOp
    return Node(inp_var, op, name='matmul_op')

  @property
  def T(self):
    inp_var = (self,)
    op = TransposeOp
    return Node(inp_var, op, name='transpose_op')

def abs(node):
  return Node((node,), AbsOp, name='abs_op')

def exp(node):
  return Node((node,), ExpOp, name='exp_op')

def log(node):
  return Node((node,), LogOp, name='log_op')

def inv(node):
  return Node((node,), InverseOp, name='inverse_op')

def det(node):
  return Node((node,), DetOp, name='det_op')

def relu(node):
  return Node((node,), ReluOp, name='relu_op')

def softmax(node):
  return Loss((node,), LogSoftmaxOp, name='softmax_op')

def transpose(node, axes=None):
  if axes:
    param['axes'] = axes
    return Node((node,), TransposeOp, param, name='transpose_op')
  else:
    return Node((node,), TransposeOp, name='transpose_op')

def tr(node):
  return Node((node,), TraceOp, name='trace_op')

def trmean(node):
  return Node((node,), TraceMeanOp, name='trace_mean_op')

def trsum(node):
  return Node((node,), TraceSumOp, name='trace_sum_op')

def sum(node):
  return Node((node,), SumOp, name='sum_op')

def mean(node):
  return Node((node,), MeanOp, name='mean_op')

class Inp(Node):

  def __init__(self, name, shape):
    super().__init__()
    self.val = 0
    self.name = name
    self.shape = shape
    return

  def forward(self, inp_dict={}):
    return inp_dict[self.name]

  def update(self, prev_grad=1, **kwargs):
    return

class Var(Node):
  
  def __init__(self, shape=(), name='Var'):
    super().__init__()
    self.val = np.random.normal(0, 1, shape)
    self.grad = 0
    self.name = name
    return

  def update(self, prev_grad=1):
    if len(np.shape(prev_grad)) > 0:
      prev_grad = shape_back(self.val, prev_grad)
    self.grad += prev_grad
    return

  def forward(self, inp_dict={}):
    return self.val.copy()

  def apply_gradient(self, lr):
    self.val = self.val - lr * np.clip(self.grad, -2, 2)
    self.grad = 0
    return


class Eye(Node):

  def __init__(self, n=1):
    super().__init__()
    self.n = n
    self.val = np.eye(n)*np.random.normal(0, 1, (n,n))
    self.grad = 0

  def update(self, prev_grad=1):
    prev_grad = shape_back(self.val, prev_grad)
    self.grad += np.eye(self.n)*prev_grad
    return

  def forward(self, inp_dict={}):
    return self.val.copy()

  def apply_gradient(self, lr):
    self.val = self.val - lr * np.clip(self.grad, -2, 2)
    self.grad = 0
    return



class Low(Node):

  def __init__(self, shape=()):
    super().__init__()
    self.val = np.tril(np.random.normal(0, 1, shape))
    self.grad = 0

  def update(self, prev_grad=1):
    if len(np.shape(prev_grad)) > 0:
      prev_grad = shape_back(self.val, prev_grad)
    self.grad += prev_grad
    return

  def forward(self, inp_dict={}):
    return self.val.copy()

  def apply_gradient(self, lr):
    self.val = self.val - lr * np.clip(np.tril(self.grad), -2, 2)
    self.grad = 0
    return

class Loss(Node):

  def __init__(self, inp_var=None, op=None, param=None, name='loss'):
    super().__init__(inp_var, op, param, name)

  def forward(self, inp_dict={}, labels=None):
    inp_val = [v.forward(inp_dict) if isinstance(v, Node) else v for v in self.inp_var]
    self.tmp_val = tuple(inp_val)
    self.op_val = self.op.run(self.tmp_val, self.param)
    self.loss_val = self.op.compute_loss(self.op_val, labels)
    return

  def update(self, labels=None):
    grad = self.op.compute_grad(labels, self.tmp_val, self.param)
    self.grad = grad
    for (v, g) in zip(self.inp_var, grad):
      if isinstance(v, Loss):
        v.update(labels)
      elif isinstance(v, Node):
        v.update(g)
    return
    
if __name__ == '__main__':
  arr = np.array([1, 2, 3, 4, 5])
  a = np.array([5, 4, 3, 2, 1])
  b = np.array([1, 2, 3, 4, 5])
  c = a@b
  print(b)
  print(c)
  a = np.array([[1, 2], [3, 4]])
  b = np.array([[2, 3], [4, 5]])
  c = a@b
  print(c)
  a = Var(shape=(5,))
#a = Var(shape=())
#ret = add(sub(mul(2, mul(a, a)) , mul(7, a)) , 3)
#ret = sub(mul(2, mul(a, a)), mul(7, a))
  ret = (2*a*a-7*a+3)**2
#ret = a**2

  for i in range(100):
    val = ret.forward()
    ret.update(0.05)
    print(val)
  print('-----var------')
  print(a.val)
  print(isinstance(a, Node))
  print(isinstance(a, Var))

  b = Var(shape=(5,))
  b_val = b.val
  A = Var(shape=(5,5))
  A_val = A.val
  print(A_val)
  ret = b@A@b
  val = ret.forward()
  print('forward val and compute val')
  print(val, b_val@A_val@b_val)
  ret.update(0)
  print(b.grad)
  print((A_val+A_val.T)@b_val)

  A = Low(shape=(3,3))
  ret = 1/2*log(abs(det(A@A.T)))
  val = ret.forward()
  ret.update(0)
  print(A.grad)
  print(np.linalg.inv(A.val.T))
    
