import numpy as np
import variational as vi
import grad as g


def gaussian():
  s = g.Inp(name='s', shape=(2,))
#v, mu, half_sigma = vi.full_gauss(s, 2)
#var_param = [mu, half_sigma]
  v, mu, log_sigma = vi.mean_gauss(s, 2)
  var_param = [mu, log_sigma]
  x = g.Inp(name='x', shape=(2,))
  x_sigma = np.array([[2, 3], [3, 5]])
#x_sigma = np.array([[2, 0], [0, 5]])
#log_pdf = g.mean(g.tr(1/2*v@v.T)+g.tr(1/2*(x-v)@np.linalg.inv(x_sigma)@(x-v).T)-g.log(g.abs(g.det(half_sigma))))-1/2*g.log(g.abs(g.det(half_sigma@half_sigma.T)))
  log_pdf = g.mean(g.tr(1/2*v@v.T)+g.tr(1/2*(x-v)@np.linalg.inv(x_sigma)@(x-v).T))
  x_mu = np.array([-3, -5])
  
  data = np.random.multivariate_normal(
      mean=x_mu, cov=x_sigma,
      size=(1000,))
  lr = 0.01
  var_gs = [0]*len(var_param)
  for i in range(1000):
    np.random.shuffle(data)
    s_sample = np.random.multivariate_normal(
        mean = np.array([0, 0]),
        cov = np.array([[1, 0],[0,1]]),
        size=(10,1,))
    inp_dict = {'x':data, 's':s_sample}
    ret = log_pdf.forward(inp_dict)
    log_pdf.update()
#half_sigma.update(prev_grad=np.linalg.inv(-L_val.T))
    log_sigma.update(prev_grad=-np.ones(2))
    for k, var in enumerate(var_param):
      var_lr, var_gs[k] = vi.learning_rate(i, var.grad, lr, var_gs[k])
      var.apply_gradient(lr)
    if (i+1)%100 == 0:
      print(ret)
      print(mu.val)
#L_val = half_sigma.val
#print(L_val@L_val.T)
    if (i+1)%500 == 0:
      lr /= 2


if __name__ == '__main__':
  gaussian()
