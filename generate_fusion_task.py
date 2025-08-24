import random

ops = [
    "linear, relu, dropout",
    "add, layernorm",
    "convolution, bias, relu",
    "softmax, dropout",
    "sigmoid, multiply",
    "add, multiply, add",
    "layernorm, dropout",
    "tanh, multiply",
    "sinc, add, relu",
    "rsqrt, multiply",
    "pow, multiply",
    "log, divide",
    "exp, sum, divide",
    "subtract, pow, mean",
    "add, gelu",
    "multiply, add, tanh",
    "linear, swish",
    "rmsnorm, add, dropout",
    # Additional operations
    "matmul, add, softplus",
    "concat, linear, silu",
    "avgpool, flatten",
    "maxpool, batchnorm, elu",
    "cosine, multiply, add",
    "abs, log1p, scale",
    "clamp, normalize",
    "interpolate, add",
    "reshape, transpose",
    "einsum, layernorm",
    "attention, add, dropout"
]

# Select 2-4 random sequences and join
result = ", ".join(random.sample(ops, random.randint(3, 4)))
print(result)