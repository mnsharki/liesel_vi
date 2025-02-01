from typing import Dict
import copy

import jax.numpy as jnp

class LieselInterface:
    def __init__(self, model):
        self.model = model
        #self.model.auto_update = False #for minibatching later


    def get_params(self) -> Dict[str, jnp.ndarray]:
        params = {}
        for pname, var in self.model.vars.items():
            params[pname] = var.value
        return params


    #probably not necessay anymore 
    def set_params(self, param_values: Dict[str, jnp.ndarray]):
        for pname, val in param_values.items():
            self.model.vars[pname].value = val


    def compute_log_prob(self, param_values: Dict[str, jnp.ndarray]) -> float:

        model_copy = copy.deepcopy(self.model)
        
        for pname, value in param_values.items():
            if pname in model_copy.vars:
                model_copy.vars[pname].value = value
            else:
                raise KeyError(f"Parameter {pname} not part of the modell.")
        
        model_copy.update()
        
        return model_copy.log_prob