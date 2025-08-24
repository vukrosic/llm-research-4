import random

ops = [
    "linear",
    "relu", 
    "add",
    "layernorm",
    "convolution",
    "bias",
    "softmax",
    "sigmoid",
    "multiply",
    "tanh",
    "sinc",
    "rsqrt",
    "pow",
    "log",
    "divide",
    "exp",
    "sum",
    "subtract",
    "mean",
    "gelu",
    "swish",
    "rmsnorm",
    "matmul",
    "softplus",
    "concat",
    "silu",
    "avgpool",
    "flatten",
    "maxpool",
    "elu",
    "cosine",
    "abs",
    "log1p",
    "scale",
    "clamp",
    "normalize",
    "interpolate",
    "reshape",
    "transpose",
    "einsum"
]

# Select 2-4 random operations and join
result = ", ".join(random.sample(ops, random.randint(3, 4)))
print(result)