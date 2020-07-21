import numpy as np
import grad as g


def sample(log_prob, var, L, M, e, inp_dict):
  
  n = var.val.shape[-1]
  sample_array = np.zeros((M, n))
  for m in range(M):
    old_val = var.val
    r0 = np.random.multivariate_normal(
        mean = np.zeros(n),
        cov = np.eye(n),
        size = ())
    r = r0.copy()
    old_log_prob = log_prob.forward(inp_dict)
    print(old_log_prob)
    log_prob.update()
    for i in range(L):
      r -= e/2*var.grad
      var.val += e*r
      var.grad = 0
      new_log_prob = log_prob.forward(inp_dict)
      log_prob.update()
      r -= e/2*var.grad
    var.grad = 0
    acc_prob = min(1, np.exp(new_log_prob-1/2*r@r-old_log_prob+1/2*r0@r0))
    if np.random.uniform() > acc_prob:
      var.val = old_val
    sample_array[m] = var.forward()
  return sample_array


if __name__ == '__main__':
  u = g.Var(shape=(2,))
  x_sigma = np.array([[2,3], [3,5]])
  x = g.Inp(name='x', shape=(2,))
  data = np.random.multivariate_normal(
      mean=np.array([-3,-5]), cov=x_sigma,
      size=(1000,))
  log_prob = 1/2*u@u.T + g.tr(1/2*(x-u)@np.linalg.inv(x_sigma)@(x-u).T)
  inp_dict = {'x':data}
  L = 100
  M = 100
  e = 1e-3
  sample_array = sample(log_prob, u, L, M, e, inp_dict)
  print(np.mean(sample_array, axis=0))
  
      
