import numpy as np

# 变量类
class Variable:
    def __init__(self, data):
        self.data = data

# 函数类
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    # 前向传播
    def forward(self, x):
        raise NotImplementedError()

# 平方函数类
class Square(Function):
    def forward(self, x):
        return x ** 2

# 数值微分
# 可以用数值微分的结果来检验反向传播的正确性，也叫梯度检验
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)