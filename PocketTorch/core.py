import numpy as np

# 变量类
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) # 如果梯度没有计算过，则初始化为1

        funcs = [self.creator]

        while funcs:
            f = funcs.pop() # 获取创建函数
            x, y = f.input, f.output # 获取输入和输出变量
            x.grad = f.backward(y.grad) # 反向传播，根据输出变量的梯度计算输入变量的梯度
            
            if x.creator is not None:
                funcs.append(x.creator) # 将输入变量的创建函数添加到队列中

# 函数类
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self) # 设定输出变量的创建函数
        self.input = input # 保存输入变量
        self.output = output # 保存输出变量
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

# 平方函数
def square(x):
    return Square()(x)

# 指数函数类    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, dout):
        x = self.input.data
        dout = np.exp(x) * dout
        return dout

# 指数函数  
def exp(x):
    return Exp()(x)

# 数值微分
# 可以用数值微分的结果来检验反向传播的正确性，也叫梯度检验
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()

print(x.grad)
