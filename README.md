# bayesian method

##variational inference
  a naive implementation of [Automatic Differentiation Variational Inference](https://dl.acm.org/doi/pdf/10.5555/3122009.3122023)
  - grad.py
  实现自动求导，与repo AutoGrad/grad.py 相同
  - variational.py
  定义Variational Inference所需的基本模块
  - variational_test.py
  测试文件，预测一个多元高斯分布的均值

##Hamiltonian Monte Carlo
  - hmc.py
  a naive implementation of [Hamiltonian Monte Carlo](https://arxiv.org/pdf/1206.1901.pdf)

##Variational Dropout
  - variatonal_dropout.py
  a simplify implementation of [Variational Dropout Sparsifies Deep Neural Networks](https://arxiv.org/pdf/1701.05369.pdf)
  50% parameters reductoins of network Lenet-300-100 in 60 iterations
