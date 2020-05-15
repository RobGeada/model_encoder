from collections import namedtuple
import numpy as np
import torch.nn as nn

try:
    import graphviz
    viz = True
except ImportError:
    print("Graphviz not installed; cell visualization functionality disabled")
    viz = False

# === HELPERS ==========================================================================================================
# calculate padding based on stride, kernel size, and dilation.
# ASSUMES EVEN SPATIAL DIMENSIONALITY!
def padsize(s=1, k=3, d=1):
    pad = np.ceil((k * d - d + 1 - s) / 2)
    return int(pad)


# print out reference guide to building models from a particular set of operations
def build_operation_set(*ops):
    operation_set = ops
    print("{:>2} | {:^15} | {:^3} | {:^3} | {:^3} |".format(
        'i',
        'Name',
        '*C',
        '*H',
        '*W'))
    print('----------------------------------------')
    for i, op in enumerate(operation_set):
        print("{:>2} | {:<15} | {:^3} | {:^3} | {:^3} |".format(
            i,
            op.name,
            op.mod[1],
            op.mod[2],
            op.mod[3],
        ))
    return operation_set


# === OPERATIONS =======================================================================================================
# build 2d grouped convolution from input channels, output channels, kernel_size, and stride
# just a handy shortcut to avoid the lengthy function call each time
def conv2d(c_in, c_out, k, s=1):
    return nn.Conv2d(c_in,
                     c_out,
                     kernel_size=k,
                     stride=s,
                     padding=padsize(k=k, s=s),
                     groups=c_in,
                     bias=False)


# operation object
Operation = namedtuple('Operation', ['name', 'function', 'mod'])


# build operation object
def build_operation(name, function, mod=None):
    if mod is None:
        mod = [1, 1, 1]
    mod = [1] + mod
    return Operation(name=name, function=function, mod=mod)


# === CELL VISUALIZER ==================================================================================================
def cell_visualizer(cell):
    if not viz:
        return None
    G = graphviz.Digraph()
    for key, val in cell.items():
        a, b = str(key[0]), str(key[1])
        G.node(a)
        G.node(b)
        for op in val:
            G.edge(a,b,label=op)
    return G


# === OUTPUT FUNCTIONS =================================================================================================
# sample classification output module
class Classifier(nn.Module):
    def __init__(self, dim, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(int(dim[1]), output_size)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


# sample regression output module
class Regressor(nn.Module):
    def __init__(self, dim, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(int(dim[1]), output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return self.sigmoid(x)


# convert output module into curried function for input into the model encoder
def build_output(output_type, output_size):
    return lambda dim: output_type(dim, output_size=output_size)


# === SOME COMMON OPERATIONS PROVIDED ==================================================================================
Identity   = build_operation('Identity',    lambda c: nn.Sequential())
ReLU       = build_operation('ReLU',        lambda c: nn.ReLU())
BatchNorm  = build_operation('BatchNorm',   lambda c: nn.BatchNorm2d(c, affine=True))
Conv1x1    = build_operation('Conv_3x3',    lambda c: conv2d(c, c, k=1))
Conv3x3    = build_operation('Conv3x3',     lambda c: conv2d(c, c, k=3))
MaxPool3x3 = build_operation('Max_Pool3x3', lambda c: nn.MaxPool2d(3, stride=1, padding=padsize(s=1)))


