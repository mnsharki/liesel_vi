import jax.numpy as jnp
from typing import Dict


class LieselInterface:
    def __init__(self, model):
        self.model = model
        self.full_data = {var_name: var.value.copy() for var_name, var in model.vars.items() if var.observed}

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
    
    def subset_obs(self, obs_idx: jnp.ndarray):

        for var_name, var in self.model.vars.items():
            if var.observed:
                var.value = jnp.take(self.full_data[var_name], obs_idx, axis=0)


    def reset_obs(self):

        for var_name, var in self.model.vars.items():
            if var.observed:
                var.value = self.full_data[var_name]
