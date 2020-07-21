import numpy as np
import grad as g

def learning_rate(i, g, lr, sp=1):
  if i == 0:
    sn = g*g
  else:
    sn = 0.1*g*g+0.9*sp
  ro = lr*pow(i+1, -0.5+1e-16)/(1+np.sqrt(sn))
  return ro, sn

def one_gauss():
  mu = g.Var(())
  sigma = g.Var(())
  return mu, sigma

def mean_gauss(s, n=1):
  mu = g.Var((n,))
  log_sigma = g.Var((n,))
  half_sigma = g.exp(log_sigma)
  v = s*half_sigma+mu
  return v, mu, log_sigma

def full_gauss(s, n=1):
  mu = g.Var((n,))
  half_sigma = g.Low(shape=(n,n))
  v = s@half_sigma.T+mu
  sigma = half_sigma@half_sigma.T
  return v, mu, half_sigma
  
