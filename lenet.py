import numpy as np
import grad as g

x = g.Inp(name='x', shape=(None, 28*28))
w1 = g.Var(name='w1', shape=(28*28, 300))
w2 = g.Var(name='w2', shape=(300, 100))
w3 = g.Var(name='w3', shape=(100, 10))

var = [w1, w2, w3]

logits = g.relu(g.relu(x@w1)@w2)@w3
loss = g.softmax(logits)

iterations = 100
batch_size = 64

train_x = np.load('/Users/gyc/Machine Learning/data/mnist/train_images.npy')
train_labels = np.load('/Users/gyc/Machine Learning/data/mnist/train_labels.npy')
test_x = np.load('/Users/gyc/Machine Learning/data/mnist/test_images.npy')
test_labels = np.load('/Users/gyc/Machine Learning/data/mnist/test_labels.npy')

train_size = train_x.shape[0]
train_x = train_x.reshape(train_size, -1)/255
test_size = test_x.shape[0]
test_x = test_x.reshape(test_size, -1)/255

lr = 0.01/batch_size

for _ in range(iterations):
  idx = np.arange(train_size)
  k = train_size//64
  np.random.shuffle(idx)
  loss_val = 0
  for j in range(k):
    inp_idx = idx[j*batch_size:(j+1)*batch_size]
    inp_x = train_x[inp_idx]
    inp_labels = train_labels[inp_idx]

    loss.forward(inp_dict={'x':inp_x}, labels=inp_labels)
    loss_val += loss.loss_val
    loss.update(labels=inp_labels)
    
    for v in var:
      v.apply_gradient(lr)

  print('iterations %d loss %f' %(_, loss_val))

loss.forward(inp_dict={'x':test_x}, labels=test_labels)
predict_val = loss.op_val
predict_id = np.argmax(predict_val, 1)
test_id = np.argmax(test_labels, 1)
acc = np.mean(predict_id == test_id)
print(acc)
print('test accuracy %f' % acc)

'''
for v in var:
  np.save(v.name+'.npy', v.val)
'''
