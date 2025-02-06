from typing import Dict
import copy

import jax
import jax.numpy as jnp

class LieselInterface:
    def __init__(self, model):
        self.model = model


    # def get_params(self) -> Dict[str, jnp.ndarray]:
    #     params = {}
    #     for pname, var in self.model.vars.items():
    #         params[pname] = var.value
    #     return params

    def get_params(self) -> Dict[str, jnp.ndarray]: #more clean
        params = {pname: var.value for pname, var in self.model.vars.items()}
        return params

    #probably not necessay anymore 
    def set_params(self, param_values: Dict[str, jnp.ndarray]):
        for pname, val in param_values.items():
            self.model.vars[pname].value = val


    def compute_log_prob(self, param_values: Dict[str, jnp.ndarray], dim_data, rng_key,
                         batch_size = None) -> float: #: Optional[Dict[str, jnp.ndarray]]

        model_copy = copy.deepcopy(self.model)
        model_copy.auto_update = False

        for pname, value in param_values.items():
            if pname in model_copy.vars:
                model_copy.vars[pname].value = value
            else:
                raise KeyError(f"Parameter {pname} not part of the model.")
            
        if batch_size is None:
            model_copy.update()
            return model_copy.log_prob, rng_key
        else:
            model_copy, rng_key = self._subset_data(model_copy, batch_size, rng_key)
            model_copy.update()
            
            scale = dim_data / batch_size
            return scale * model_copy.log_prob, rng_key
            

    def _subset_data(self, model, batch_size, rng_key):

        batch_size = int(batch_size) if isinstance(batch_size, (int, float)) else batch_size

        
        rng_key, subset_rng = jax.random.split(rng_key)
        for var_name, var in model.vars.items():
            if getattr(var, "observed", True):
                indices = jax.random.choice(
                    subset_rng, var.value.shape[0], (batch_size,), replace=True
            )
                
                var.value = var.value[indices]  

        return model, rng_key
    