import jax
import jax.numpy as jnp
import optax
from typing import Dict, List
from tensorflow_probability.substrates import jax as tfp
from jax.flatten_util import ravel_pytree
import jax.tree_util 

from .interface import LieselInterface
tfd = tfp.distributions


class Optimizer:
    def __init__(
        self,
        seed: int,
        n_epochs: int,
        model_interface: LieselInterface,
        latent_variables: List[Dict],
    ):
        self.seed = seed
        self.n_epochs = n_epochs
        self.model_interface = model_interface
        self.latent_vars_config = latent_variables
        self.rng_key = jax.random.PRNGKey(self.seed)

        self.variational_dists = self._init_variational_dists()
        self.flat_params, self._unravel_fn = ravel_pytree(self.variational_dists)

        self.opt_state, self.optimizer = self._init_optimizer()
        self.elbo_values = []
        


    def _init_variational_dists(self):


        variational_dists = {
            pname: config["distribution"]
            for config in self.latent_vars_config
            for pname in config["names"]
        }

        return variational_dists 
    
    def _init_transform_dict(self):
        optim_dict = {
                    pname: config["optimizer_chain"]
                    for config in self.latent_vars_config
                    for pname in config["names"]
                }
        return optim_dict


    def _init_optimizer(self):
        
        def label_fn(param_tree):
            return {k: k for k in param_tree}
        
        param_tree = self._unravel_fn(self.flat_params)
        #flat_params = self.variational_dists #, unravel_fn = ravel_pytree(self.variational_dists)
        #self._unravel_fn = unravel_fn

        #base_optimizer = self.intermediate_optimizer 

        optim_dict = self._init_transform_dict()
        tx = optax.multi_transform(optim_dict, label_fn) #hier nicht flat params sondern dictionar mit optimizers

        #opt_state = base_optimizer.init(flat_params)
        opt_state = tx.init(param_tree)
        return opt_state, tx #base_optimizer



    def fit(self):
        @jax.jit
        def step(flat_params, opt_state, rng_key):
            param_tree = self._unravel_fn(flat_params)
            (loss_val, rng_key), grads = jax.value_and_grad(
                lambda p, key: self._elbo(p, key),
                has_aux=True
            )(param_tree, rng_key)

            updates, new_opt_state = self.optimizer.update(grads, opt_state, param_tree)
            new_param_tree = optax.apply_updates(param_tree, updates)
            new_flat_params, _ = ravel_pytree(new_param_tree)
            return new_flat_params, new_opt_state, loss_val, rng_key

        flat_params = self.flat_params #flat_params, unravel_fn = ravel_pytree(self.variational_dists)
        opt_state = self.opt_state
        rng_key = self.rng_key

        for epoch in range(self.n_epochs):
            flat_params, opt_state, loss_val, rng_key = step(flat_params, opt_state, rng_key)
            self.elbo_values.append(-loss_val.item())
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}, ELBO: {-loss_val:.4f}")

        self.opt_state = opt_state
        self.rng_key = rng_key
        self.flat_params = flat_params  #self.variational_dists = unravel_fn(flat_params)


    def _elbo(self, param_tree, rng_key): #flat_params

        variational_dists = param_tree #flat_params #variational_dists = self._unravel_fn(flat_params) 
        

        num_samples = 32

        rng_keys = jax.random.split(rng_key, num_samples)

        def single_sample_elbo(rng_key):
            samples, log_det_jac, _ = self._sample_variational(variational_dists, rng_key)  #variational_params
            log_likelihood = self.model_interface.compute_log_likelihood(samples) + log_det_jac
            log_prior = self.model_interface.compute_log_prior(samples)
            return log_likelihood + log_prior 

        elbo_samples = jax.vmap(single_sample_elbo)(rng_keys)

        elbo = jnp.mean(elbo_samples)
            
        #Apparently one time calculation of entropy is enough, bc deterministic
        entropy = self._compute_entropy(variational_dists) 
        elbo += entropy

        return -elbo, rng_key
    

#Considerations: Bijectors of add_latent_variable only affect the samples and their domain 
#The parameters of distribution need to be of a specific domain 
#However, the log_det_jac is calculated in the transformed space, so the bijector should be applied to the log_det_jac a well
    def _sample_variational(self, variational_dists, rng_key):
        samples = {}
        log_det_jac = 0.0

        name_to_transform = {}
        for config in self.latent_vars_config:
            transform = config.get("transform", None)
            for pname in config["names"]:
                name_to_transform[pname] = transform

        for pname, dist_obj in variational_dists.items():
            if dist_obj.reparameterization_type == tfd.FULLY_REPARAMETERIZED:
                rng_key, subkey = jax.random.split(rng_key)
                z = dist_obj.sample(seed=subkey)
            
            else: 
                NotImplementedError("Only fully reparameterized distributions are supported so far.")


            transform_spec = name_to_transform[pname]
            if transform_spec is None:
                z_transformed = z
                ldj = 0.0
            elif callable(transform_spec) and not hasattr(transform_spec, "forward"):
                z_transformed, ldj = transform_spec(z)
            elif hasattr(transform_spec, "forward") and hasattr(transform_spec, "forward_log_det_jacobian"):
                event_ndims = 1 if z.ndim == 1 else 0
                z_transformed = transform_spec.forward(z)
                ldj = transform_spec.forward_log_det_jacobian(z, event_ndims=event_ndims)
                if ldj.ndim > 0:
                    ldj = jnp.sum(ldj)
            else:
                raise ValueError(f"Unrecognized transform for {pname}")

            log_det_jac += ldj
            samples[pname] = z_transformed

        return samples, log_det_jac, rng_key

    def _compute_entropy(self, variational_dists):
        total_entropy = 0.0
        for dist in variational_dists.values():
            total_entropy += jnp.sum(dist.entropy())

        return total_entropy



def _tfp_dist_flatten_fn(dist_obj):
    return (), None

def _tfp_dist_unflatten_fn(aux, children):
    return aux  

jax.tree_util.register_pytree_node(
    tfp.distributions.Distribution,
    _tfp_dist_flatten_fn,
    _tfp_dist_unflatten_fn
)