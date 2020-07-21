import numpy as np
import grad as g

def sigmoid(x):
  return 1/(1+np.exp(-x))

'''regularization term i.e. DL term
DL = 0.5*log(1+1/sigma) - k1*sigmoid(k2 + k3 * logsigma) + C
derivative: -0.5*1/(1+logsigma) - k1*sigmoid*(1-sigmoid)*k3
'''

k1 = 0.63576
k2 = 1.87320
k3 = 1.48695
C = -k1

class SparseVDLayer():

  def __init__(self, in_layer, shape=(), batch_size=64, activation=None, name='sparse_layer', initial_value=None):
    if initial_value is not None:
      self.shape = initial_value.shape
      self.size = initial_value.size
      self.mu = initial_value
    else:
      self.shape = shape
      self.size = np.prod(shape)
      self.mu = np.random.normal(size=shape)
    self.name = name
    self.log_sigma = np.ones(self.shape)*-5
    self.op = activation
    self.rand_shape = (batch_size, self.shape[1])
    self.rand = np.random.normal(size=self.rand_shape)
    self.in_layer = in_layer
    self.grad = 0
    self.test = False
    return
    
  def update(self, prev_grad=1, lr=0.001):
    '''
    if len(np.shape(prev_grad)) > 0:
      prev_grad = shape_back(self.val, prev_grad)
    '''
    if self.op:
      grad, = self.op.compute_grad(prev_grad, (self.logits,))
      self.grad = grad
    else:
      grad = prev_grad
      self.grad = prev_grad
    grad1 = grad@self.mask_mu.T
    grad2 = (grad*self.rand*0.5/self.std)@((np.exp(self.log_alpha)*self.mu*self.mu).T)*2*self.inp
    next_grad = grad1+grad2
    self.in_layer.update(prev_grad=next_grad, lr=lr)
    self.apply_gradient(lr=lr)
    
    return
    
  def forward(self, inp_dict={}):

    self.inp = self.in_layer.forward(inp_dict)
    log_alpha = self.log_sigma - np.log(self.mu*self.mu)
    self.clip_mask = (log_alpha < 6) * (log_alpha > -6)
    self.log_alpha = np.clip(log_alpha, -6, 6)
    self.mask_mu = self.mu * (log_alpha <= 0.7)
    mean = self.inp@self.mask_mu
    self.std = np.sqrt(np.dot(self.inp*self.inp, np.exp(self.log_alpha)*self.mu*self.mu)+1e-8)
    if self.test:
      self.logits = self.inp@self.mu
    else:
      self.logits = mean + self.std*self.rand

    if self.op:
      ret = self.op.run((self.logits,))
      return ret
    else:
      return self.logits
    

  def apply_gradient(self, lr):
    log_alpha = self.log_sigma - np.log(self.mu*self.mu)
    mu_grad = (self.inp.T@self.grad)*(log_alpha <= 0.7)
    log_sigma_grad = (self.inp*self.inp).T @ (self.grad*self.rand*0.5/self.std) * (np.exp(self.log_alpha)*self.mu*self.mu) #* self.clip_mask

    '''regularization term grad'''
#log_alpha = self.log_sigma - 2*np.log(np.abs(self.mu))
    sig = sigmoid(k2 + k3 * self.log_alpha)
#    reg_grad = -k1*sig*(1-sig)*k3 - 0.5/(np.exp(self.log_alpha)+1)
    reg_grad = -0.5*(log_alpha <= 0.7)
    reg_mu_grad = -reg_grad*0.5/self.mu
    
    self.reg_val = k1 * sig - 0.5*np.log(1+np.exp(-self.log_alpha)) + C

    mu_grad = (mu_grad + 0.1*reg_mu_grad)*(log_alpha<=2)
    log_sigma_grad = log_sigma_grad + 0.1*(-0.5)*(log_alpha <= 0.7)

    self.mu -= lr * mu_grad
    self.log_sigma -= lr * log_sigma_grad

    self.fresh()
    return

  def set_reg_param(self, param):
    self.reg_param = param
    return

  def fresh(self):
    self.rand = np.random.normal(size=self.rand_shape)
    self.inp = None
    self.logits = None
    self.std = None
    self.log_alpha = None
    self.grad = 0
    return

  def sparse(self):
    self.test = True
    log_alpha = self.log_sigma - 2*np.log(np.abs(self.mu))
    self.mu = self.mu * (log_alpha <= 0.7)
    return

iterations = 100
batch_size = 128

    
x = g.Inp(name='x', shape=(batch_size, 28*28))
w1_init = np.load('w1.npy')
w2_init = np.load('w2.npy')
w3_init = np.load('w3.npy')
'''
w1 = SparseVDLayer(x, batch_size=batch_size, name='w1', shape=(28*28, 300), activation=g.ReluOp, initial_value=w1_init)
w2 = SparseVDLayer(w1, batch_size=batch_size, name='w2', shape=(300, 100), activation=g.ReluOp, initial_value=w2_init)
w3 = SparseVDLayer(w2, batch_size=batch_size, name='w3', shape=(100, 10), initial_value=w3_init)
'''

w1 = SparseVDLayer(x, batch_size=batch_size, name='w1', shape=(28*28, 300), activation=g.ReluOp, initial_value=None)
w2 = SparseVDLayer(w1, batch_size=batch_size, name='w2', shape=(300, 100), activation=g.ReluOp, initial_value=None)
w3 = SparseVDLayer(w2, batch_size=batch_size, name='w3', shape=(100, 10), initial_value=None)

var = [w1, w2, w3]

train_x = np.load('/Users/gyc/Machine Learning/data/mnist/train_images.npy')
train_labels = np.load('/Users/gyc/Machine Learning/data/mnist/train_labels.npy')
test_x = np.load('/Users/gyc/Machine Learning/data/mnist/test_images.npy')
test_labels = np.load('/Users/gyc/Machine Learning/data/mnist/test_labels.npy')

train_size = train_x.shape[0]
train_x = train_x.reshape(train_size, -1)/255
test_size = test_x.shape[0]
test_x = test_x.reshape(test_size, -1)/255


reg_param = train_size//batch_size

lr = 1e-6*reg_param

for v in var:
  v.set_reg_param(1.0/reg_param)

for _ in range(iterations):
  idx = np.arange(train_size)
  k = reg_param
  np.random.shuffle(idx)
  loss_val = 0
  reg_val = 0
  for j in range(k):
    inp_idx = idx[j*batch_size:(j+1)*batch_size]
    inp_x = train_x[inp_idx]
    inp_labels = train_labels[inp_idx]

    logits = w3.forward(inp_dict={'x':inp_x})
    grad = g._softmax(logits)-inp_labels
    w3.update(grad, lr=lr)
    loss_val += -np.sum(np.log(g._softmax(logits)+1e-8)*inp_labels)
    

    reg_val += w1.reg_val

  print('iterations %d loss %f reg_val %f' %(_, loss_val, np.mean(reg_val)))

for v in var:
  v.sparse()

predict_val = w3.forward(inp_dict={'x':test_x})
predict_id = np.argmax(predict_val, 1)
test_id = np.argmax(test_labels, 1)
acc = np.mean(predict_id == test_id)
print(acc)
print('test accuracy %f' % acc)

for v in var:
  np.save(v.name+'_sparse.npy', v.mu)

print(np.sum(w1.mu != 0)/np.sum(w1_init != 0), np.sum(np.abs(w1.mu))/np.sum(np.abs(w1_init)))
print(np.sum(w2.mu != 0)/np.sum(w2_init != 0), np.sum(np.abs(w2.mu))/np.sum(np.abs(w2_init)))
print(np.sum(w3.mu != 0)/np.sum(w3_init != 0), np.sum(np.abs(w3.mu))/np.sum(np.abs(w3_init)))
