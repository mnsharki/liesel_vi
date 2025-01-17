import jax
import jax.numpy as jnp
import optax
from typing import Dict, List
from tensorflow_probability.substrates import jax as tfp
from jax.flatten_util import ravel_pytree

from .interface import LieselInterface
tfd = tfp.distributions


class Optimizer:
    def __init__(
        self,
        seed: int,
        n_epochs: int,
        lr: float,
        model_interface: LieselInterface,
        latent_variables: List[Dict]
    ):
        self.seed = seed
        self.n_epochs = n_epochs
        self.lr = lr
        self.model_interface = model_interface
        self.latent_vars_config = latent_variables
        self.rng_key = jax.random.PRNGKey(self.seed)
        self.variational_params = self._init_variational_params()
        self.opt_state, self.optimizer = self._init_optimizer()
        self.elbo_values = []

    def _init_variational_params(self): #-> Dict[str, Dict[str, jnp.ndarray]]:
        # variational_params = {}


        variational_dists = {
            pname: config["distribution"]
            for config in self.latent_vars_config
            for pname in config["names"]
        }
        # for config in self.latent_vars_config:
        #     dist = config["distribution"]
        #     names = config["names"]
        #     for pname in names:
        #         variational_params[pname] = dist
        #         if isinstance(dist, tfd.Normal):
        #             init_mu = jnp.asarray(dist.loc)
        #             init_log_sigma = jnp.log(jnp.asarray(dist.scale))
        #             variational_params[pname] = {"mu": init_mu, "log_sigma": init_log_sigma}
        #         elif isinstance(dist, tfd.MultivariateNormalFullCovariance):
        #             cov = jnp.asarray(dist.covariance())
        #             chol = jnp.linalg.cholesky(cov)
        #             chol_unconstr = jnp.where(
        #                 jnp.eye(chol.shape[0], dtype=bool),
        #                 jnp.log(jnp.diag(chol)),
        #                 chol
        #             )
        #             variational_params[pname] = {
        #                 "mu": jnp.asarray(dist.loc),
        #                 "chol_unconstrained": chol_unconstr
        #             }
        #         else:
        #             raise TypeError(
        #                 f"Unsupported distribution for param '{pname}'. Use Normal or MultivariateNormalFullCovariance."
        #             )
        return variational_dists # variational_params

    def _init_optimizer(self):
        flat_params, unravel_fn = ravel_pytree(self.variational_params)
        self._unravel_fn = unravel_fn
        base_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(self.lr)
        )
        opt_state = base_optimizer.init(flat_params)
        return opt_state, base_optimizer



    def fit(self):
        @jax.jit
        def step(flat_params, opt_state, rng_key):
            (loss_val, rng_key), grads = jax.value_and_grad(
                lambda p, key: self._elbo(p, key),
                has_aux=True
            )(flat_params, rng_key)

            updates, new_opt_state = self.optimizer.update(grads, opt_state, flat_params)
            new_flat_params = optax.apply_updates(flat_params, updates)
            return new_flat_params, new_opt_state, loss_val, rng_key

        #from jax.flatten_util import ravel_pytree
        flat_params, unravel_fn = ravel_pytree(self.variational_params)
        opt_state = self.opt_state
        rng_key = self.rng_key

        for epoch in range(self.n_epochs):
            flat_params, opt_state, loss_val, rng_key = step(flat_params, opt_state, rng_key)
            self.elbo_values.append(-loss_val.item())
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}, ELBO: {-loss_val:.4f}")

        # Update final states
        self.opt_state = opt_state
        self.rng_key = rng_key
        self.variational_params = unravel_fn(flat_params)

    # def _elbo(self, flat_params, rng_key):
    #     variational_params = self._unravel_fn(flat_params)
    #     samples, log_det_jac, rng_key = self._sample_variational(variational_params, rng_key)

    #       self.model_interface.set_params(samples)
    #       log_likelihood = self.model_interface.log_likelihood + log_det_jac
    #       log_prior = self.model_interface.log_prior

    #     # log_likelihood = self.model_interface.compute_log_likelihood(samples) + log_det_jac
    #     # log_prior = self.model_interface.compute_log_prior(samples)

    #     entropy = self._compute_entropy(variational_params)
    #     elbo = log_likelihood + log_prior + entropy
    #     return -elbo, rng_key

    def _elbo(self, flat_params, rng_key):

        variational_dists = self._unravel_fn(flat_params) # variational_params = self._unravel_fn(flat_params)

        num_samples = 32

        rng_keys = jax.random.split(rng_key, num_samples)

        def single_sample_elbo(rng_key):
            samples, log_det_jac, _ = self._sample_variational(variational_dists, rng_key)  #variational_params
            log_likelihood = self.model_interface.compute_log_likelihood(samples) + log_det_jac
            log_prior = self.model_interface.compute_log_prior(samples)
            return log_likelihood + log_prior

        elbo_samples = jax.vmap(single_sample_elbo)(rng_keys)

        elbo = jnp.mean(elbo_samples)
            
        # Apparently one time calculation of entropy is enough, bc deterministic
        entropy = self._compute_entropy(variational_dists) #variational_params
        elbo += entropy

        return -elbo, rng_key



    # def _sample_variational(self, variational_params, rng_key):

    #     samples = {}
    #     log_det_jac = 0.0

    #     #map param_name -> transform_type
    #     name_to_transform = {}
    #     for config in self.latent_vars_config:
    #         transform = config.get("transform", None)
    #         for pname in config["names"]:
    #             name_to_transform[pname] = transform

    #     for pname, pvars in variational_params.items():

    #         if "log_sigma" in pvars: 
    #             mu, log_sigma = pvars["mu"], pvars["log_sigma"]
    #             rng_key, subkey = jax.random.split(rng_key)
    #             eps = jax.random.normal(subkey, mu.shape)
    #             z = mu + jnp.exp(log_sigma) * eps

    #         elif "chol_unconstrained" in pvars:

    #             mu = pvars["mu"]
    #             chol_unconstr = pvars["chol_unconstrained"]
    #             chol = self._reconstruct_chol(chol_unconstr)

    #             rng_key, subkey = jax.random.split(rng_key)
    #             eps = jax.random.normal(subkey, (mu.shape[0],))
    #             z = mu + chol @ eps

    #         else:
    #             raise ValueError(f"Unrecognized var param representation for '{pname}'.")
    #         transform_spec = name_to_transform.get(pname, None)
    #         if transform_spec == "positive":
    #             z_transformed = jnp.exp(z)
    #             log_det_jac += jnp.sum(z) 
    #             samples[pname] = z_transformed

    #         elif isinstance(transform_spec, dict) and "bounds" in transform_spec:
    #             a, b = transform_spec["bounds"]
    #             sigmoid_z = jax.nn.sigmoid(z)
    #             z_transformed = a + (b - a) * sigmoid_z
    #             log_det_jac += jnp.sum(jnp.log((b - a) * sigmoid_z * (1 - sigmoid_z)))
    #             samples[pname] = z_transformed

    #         else:
    #             samples[pname] = z

    #     return samples, log_det_jac, rng_key

    def _sample_variational(self, variational_params, rng_key):
        samples = {}
        log_det_jac = 0.0

        name_to_transform = {}
        for config in self.latent_vars_config:
            transform = config.get("transform", None)
            for pname in config["names"]:
                name_to_transform[pname] = transform

        for pname, pvars in variational_params.items():


            if "log_sigma" in pvars:
                mu, log_sigma = pvars["mu"], pvars["log_sigma"]
                rng_key, subkey = jax.random.split(rng_key)
                eps = jax.random.normal(subkey, mu.shape)
                z = mu + jnp.exp(log_sigma) * eps

            elif "chol_unconstrained" in pvars:
                mu = pvars["mu"]
                chol_unconstr = pvars["chol_unconstrained"]
                chol = self._reconstruct_chol(chol_unconstr)
                rng_key, subkey = jax.random.split(rng_key)
                eps = jax.random.normal(subkey, (mu.shape[0],))
                z = mu + chol @ eps

            else:
                raise ValueError(f"Unrecognized var param representation for '{pname}'.")


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
            else:
                raise ValueError(
                    f"For param '{pname}', transform_spec is not None/Callable/Bijector. "
                    "Got something unrecognized."
                )


            if jnp.ndim(ldj) > 0:
                ldj = jnp.sum(ldj)

            log_det_jac += ldj
            samples[pname] = z_transformed

        return samples, log_det_jac, rng_key


        

    def _reconstruct_chol(self, chol_unconstr):

        diag = jnp.exp(jnp.diag(chol_unconstr))
        chol = chol_unconstr.at[jnp.diag_indices(diag.shape[0])].set(diag)
        return chol

    def _compute_entropy(self, variational_params):

        total_entropy = 0.0
        for pname, pvars in variational_params.items():
            if "log_sigma" in pvars:
                mu = pvars["mu"]
                log_sigma = pvars["log_sigma"]
                dist_obj = tfd.Normal(mu, jnp.exp(log_sigma))
                # sum entropies if vector
                total_entropy += jnp.sum(dist_obj.entropy())

            elif "chol_unconstrained" in pvars:
                mu = pvars["mu"]
                chol_unconstr = pvars["chol_unconstrained"]
                chol = self._reconstruct_chol(chol_unconstr)

                cov = chol @ chol.T
                dist_obj = tfd.MultivariateNormalFullCovariance(mu, cov)
                total_entropy += dist_obj.entropy()

            else:
                raise ValueError(f"Entropy not defined for param '{pname}'.")
        return total_entropy