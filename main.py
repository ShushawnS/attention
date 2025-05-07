import torch
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from causal_self_attention_torch import CausalSelfAttentionTorch
from causal_self_attention_jax import CausalSelfAttentionJax

class Config:
    def __init__(self):
        self.n_embd = 128
        self.n_head = 1
        self.block_size = 256
        self.dropout = 0.0
        self.bias = True

# ensure that the params for both models are identical
def copy_params_torch_to_jax(torch_model, jax_params):
    flat_params = {}

    for (name, param) in torch_model.named_parameters():
        key_chain = name.replace(".weight", "").replace(".bias", "").split(".")
        leaf = 'bias' if 'bias' in name else 'kernel'
        val = param.detach().numpy().T if 'weight' in name else param.detach().numpy()

        # nest into flax-style dict
        d = flat_params
        for k in key_chain:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[leaf] = jnp.array(val)

    # recursively update jax_params with flat_params
    def recursive_update(d1, d2):
        for k, v in d2.items():
            if isinstance(v, dict):
                d1[k] = recursive_update(d1.get(k, {}), v)
            else:
                d1[k] = v
        return d1

    return recursive_update(jax_params, flat_params)

# init
config = Config()
modelTorch = CausalSelfAttentionTorch(config)
modelJax = CausalSelfAttentionJax(
    n_embd=config.n_embd,
    n_head=config.n_head,
    dropout=config.dropout,
    bias=config.bias,
    block_size=config.block_size,
)

B, T = 2, 10
x_torch = torch.randn(B, T, config.n_embd)
x_jax = jnp.array(x_torch.detach().numpy(), dtype=float32)
key = jax.random.PRNGKey(0)
params = modelJax.init(key, x_jax)
params = copy_params_torch_to_jax(modelTorch, params)

# run models
outputTorch = modelTorch(x_torch).detach().numpy()
outputJax = modelJax.apply(params, x_jax, deterministic=True)

# tests
print("Shape Match:", outputTorch.shape == outputJax.shape)
print("Max Abs Diff:", np.max(np.abs(outputTorch - outputJax)))
