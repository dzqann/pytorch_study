#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2020/5/12 15:44 
# @Author : DZQ
# @File : basic.py

import torch
from torch.autograd import Variable
from sklearn import datasets

# x = Variable(torch.Tensor([1]), requires_grad=True)
# w = Variable(torch.Tensor([2]), requires_grad=True)
# b = Variable(torch.Tensor([3]), requires_grad=True)
# y = w * x + b
# y.backward()
# print(x.grad)
# print(w.grad)
# print(b.grad)

# x = torch.randn(3)
# x = Variable(x, requires_grad=True)
# y = x * 2
# print(y)
# y.backward(torch.FloatTensor([1, 0.1, 0.01]))
# print(x.grad)

batch_size = 32
random = torch.randn(batch_size)
print(random)
print(random.unsqueeze(1))
x = random.unsqueeze(1)
print([x ** i for i in range(1, 4)])
print(torch.cat([x ** i for i in range(1, 4)], 1))
