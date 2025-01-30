import jax.numpy as jnp
from typing import Dict


class LieselInterface:
    def __init__(self, model):
        self.model = model

    def get_params(self) -> Dict[str, jnp.ndarray]:
        params = {}
        for param in self.model.vars.values():
            params[param.name] = jnp.asarray(param.value)
        return params

    def set_params(self, param_values: Dict[str, jnp.ndarray]):
        for pname, val in param_values.items():
            self.model.vars[pname].value = val

    # def compute_log_likelihood(self, samples: Dict[str, jnp.ndarray]) -> float:
    #     self.set_params(samples)
    #     return self.model.log_lik

    # def compute_log_prior(self, samples: Dict[str, jnp.ndarray]) -> float:
    #     self.set_params(samples)
    #     return self.model.log_prior

    def compute_log_prob(self, samples: Dict[str, jnp.ndarray]) -> float:
        self.set_params(samples)
        return self.model.log_prob