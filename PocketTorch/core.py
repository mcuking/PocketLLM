import numpy as np

# 变量类
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None

# 函数类
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input # 保存输入变量
        return output

    # 前向传播
    def forward(self, x):
        raise NotImplementedError()
    
    # 反向传播
    def backward(self, dout):
        raise NotImplementedError()

# 平方函数类
class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, dout):
        x = self.input.data
        dout = 2 * x * dout
        return dout

# 指数函数类    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, dout):
        x = self.input.data
        dout = np.exp(x) * dout
        return dout

# 数值微分
# 可以用数值微分的结果来检验反向传播的正确性，也叫梯度检验
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


A = Square()
B = Exp()
C = Square()
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)

print(x.grad)
