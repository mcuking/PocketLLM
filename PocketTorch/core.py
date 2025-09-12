import numpy as np

# 变量类
class Variable:
    def __init__(self, data):
        if data is not None and not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not np.ndarray")

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
            gys = [output.grad for output in f.outputs] # 获取输出变量的梯度
            gxs = f.backward(*gys) # 反向传播，计算输入变量的梯度
            if not isinstance(gxs, tuple):
                gxs = (gxs,) # 如果梯度不是元组，则将其转换为元组，以便统一处理

            for x, gx in zip(f.inputs, gxs): # 遍历输入变量和梯度
                x.grad = gx # 将输入变量的梯度设置为计算得到的梯度
            
                if x.creator is not None:
                    funcs.append(x.creator) # 将输入变量的创建函数添加到队列中

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# 函数类
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs] # 支持函数的多个输入
        ys = self.forward(*xs) # 计算输出
        if not isinstance(ys, tuple):
            ys = (ys,) # 如果输出不是元组，则将其转换为元组，以便统一处理
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self) # 设定输出变量的创建函数

        self.inputs = inputs # 保存输入变量
        self.outputs = outputs # 保存输出变量

        return outputs if len(outputs) > 1 else outputs[0] # 如果输出变量只有一个，则返回单个变量，否则返回元组

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
        x = self.inputs[0].data
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
        x = self.inputs[0].data
        dout = np.exp(x) * dout
        return dout

# 指数函数  
def exp(x):
    return Exp()(x)

# 加法函数类
class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1
    
# 加法函数
def add(x0, x1):
    return Add()(x0, x1)

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
