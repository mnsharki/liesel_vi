from typing import Dict
import copy

import jax.numpy as jnp

class LieselInterface:
    def __init__(self, model):
        self.model = model


    def get_params(self) -> Dict[str, jnp.ndarray]: #more clean
        params = {pname: jnp.array(var.value) for pname, var in self.model.vars.items()} #allows single int input for params
        return params


    #probably not necessay anymore 
    # def set_params(self, param_values: Dict[str, jnp.ndarray]):
    #     for pname, val in param_values.items():
    #         self.model.vars[pname].value = val


    def compute_log_prob(self, param_values: Dict[str, jnp.ndarray], rng_key, dim_data,
                         batch_size, batch_indices):

        model_copy = copy.deepcopy(self.model)
        model_copy.auto_update = False

        for pname, value in param_values.items():
            if pname in model_copy.vars:
                model_copy.vars[pname].value = value
            else:
                raise KeyError(f"Parameter {pname} not part of the model.")
            
        if batch_size is None:
            model_copy.update()
            return model_copy.log_prob
         
        else:
            model_copy= self._subset_data(model_copy, batch_indices)    
            model_copy.update()

            scale = (dim_data / batch_size)
            log_likelihood = scale * model_copy.log_lik   #Kucukelbir only scaling of likelihood  
            log_prior = model_copy.log_prior
            log_prob = log_likelihood + log_prior

            return log_prob
            

    def _subset_data(self, model, batch_indices):

        for var in model.vars.values():
            if getattr(var, "observed", True):
                var.value = var.value[batch_indices]  

        return model



    