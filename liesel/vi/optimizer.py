from typing import Dict, List


import jax
import jax.numpy as jnp
import optax
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

        self.variational_dists_class = self._init_variational_dists_class()
        self.phi = self._init_phi()
        self.fixed_distribution_params = self._init_fixed_distribution_params()

        self.initial_distributions = self._validate_and_build_distributions() 

        self.opt_state, self.optimizer = self._init_optimizer()
        self.elbo_values = []

    
    def _init_variational_dists_class(self):

        variational_dists_class = {
            pname: config["dist_class"]
            for config in self.latent_vars_config
            for pname in config["names"]
        }

        return variational_dists_class       


    def _init_phi(self):
        
        phi = {
            pname: config["phi"]
            for config in self.latent_vars_config
            for pname in config["names"]
        }

        return phi


    def _init_fixed_distribution_params(self):
        fixed_distribution_params = {
            pname: config["fixed_distribution_params"] if config["fixed_distribution_params"] is not None else {}  
            for config in self.latent_vars_config
            for pname in config["names"]
        }
        return fixed_distribution_params

    
    def _init_transform_dict(self):

        optim_dict = {
                    pname: config["optimizer_chain"]
                    for config in self.latent_vars_config
                    for pname in config["names"]
                }
        
        return optim_dict
    

    def _build_distribution(self, dist_class, phi, fixed_distribution_params):
        return dist_class(**phi, **fixed_distribution_params)
    

    def _validate_and_build_distributions(self):

        phi_keys = set(self.phi.keys())
        fixed_keys = set(self.fixed_distribution_params.keys())
        dist_keys = set(self.variational_dists_class.keys())

        if not (phi_keys == fixed_keys == dist_keys):
            raise ValueError(f"Mismatch in keys: phi_keys={phi_keys}, fixed_keys={fixed_keys}, dist_keys={dist_keys}")

        distributions = {
            key: self._build_distribution(self.variational_dists_class[key], self.phi[key], self.fixed_distribution_params[key])
            for key in phi_keys  
        }

        return distributions


    def _init_optimizer(self):

        # def label_fn(d): #more restrictive label fn
        #     return {k: k for k in d}

        def label_fn(d): #updated label fn for consistency 
            return {k: k for k in d if k in self.phi}

        
        phi_dict = self.phi

        optim_dict = self._init_transform_dict()
        tx = optax.multi_transform(optim_dict, label_fn) 
        opt_state = tx.init(phi_dict)

        return opt_state, tx


    def fit(self):
        
        @jax.jit
        def step(current_phi, opt_state, rng_key): 
    
            (loss_val, rng_key), grads = jax.value_and_grad(
                lambda p, key: self._elbo(p, key),
                has_aux=True
            )(current_phi, rng_key)

            updates, new_opt_state = self.optimizer.update(grads, opt_state, current_phi) 
            new_phis = optax.apply_updates(current_phi, updates) 
 
            return new_phis, new_opt_state, loss_val, rng_key

        phi = self.phi  
        opt_state = self.opt_state
        rng_key = self.rng_key

        for epoch in range(self.n_epochs):
            phi, opt_state, loss_val, rng_key = step(phi, opt_state, rng_key) 
            self.elbo_values.append(float(-loss_val))
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}, ELBO: {-loss_val:.4f}")

        self.phi = phi
        self.opt_state = opt_state
        self.rng_key = rng_key


    #@jax.jit
    def _elbo(self, phi, rng_key): 

        num_samples = 32

        rng_keys = jax.random.split(rng_key, num_samples)

        #@jax.jit
        def _single_sample_elbo(rng_key):
            samples, log_det_jac, log_q = self._sample_variational(phi, rng_key)  
            log_prob = self.model_interface.compute_log_prob(samples) + log_det_jac
            return log_prob - log_q

        elbo_samples = jax.vmap(_single_sample_elbo)(rng_keys)
        elbo = jnp.mean(elbo_samples)

        return -elbo, rng_key
    
    #@jax.jit
    def _sample_variational(self, phi, rng_key): 
        # you could split this function with a function sample_variational prob. useful for other purposes as well 
        samples = {}
        log_det_jac = 0.0
        log_q_z = 0.0

        name_to_transform = {}
        for config in self.latent_vars_config:
            transform = config.get("transform", None)
            for pname in config["names"]:
                name_to_transform[pname] = transform
                

        for pname, pval in phi.items(): 

            dist_obj = self._build_distribution(self.variational_dists_class[pname], 
                                                pval, 
                                                self.fixed_distribution_params[pname])


            if dist_obj.reparameterization_type == tfd.FULLY_REPARAMETERIZED:
                rng_key, subkey = jax.random.split(rng_key)
                z = dist_obj.sample(seed=subkey) 

                log_q_z += dist_obj.log_prob(z)
            
            else: 
                raise NotImplementedError("Only fully reparameterized distributions are supported so far.")


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


        return samples, log_det_jac, log_q_z