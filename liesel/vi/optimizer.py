from typing import Dict, List, Optional
import math
import jax
import jax.numpy as jnp
import optax
from tensorflow_probability.substrates import jax as tfp
import jax.tree_util
from .interface import LieselInterface

import matplotlib.pyplot as plt
import seaborn as sns

tfd = tfp.distributions

from functools import partial


class Optimizer:
    """
    Optimizer for stochastic variational inference.

    This class performs variational inference by optimizing the ELBO using gradient-based
    methods. It initializes variational distributions based on a given model interface and latent
    variable configurations, then runs an optimization loop over a specified number of epochs.
    """

    def __init__(
        self,
        seed: int,
        n_epochs: int,
        S: int,
        model_interface: LieselInterface,
        latent_variables: List[Dict],
        batch_size: Optional[int] = None, 
        patience_tol: Optional[float] = None, 
        window_size: Optional[int] = None,  
    ) -> None:
        """
        Initialize the Optimizer.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        n_epochs : int
            Number of epochs to run the optimization.
        S : int
            Number of Monte Carlo samples.
        model_interface : LieselInterface
            Interface to access the model parameters data and quantities.
        latent_variables : List[Dict]
            List of configurations for each latent variable.
        batch_size : int, optional
            Batch size for data subsetting; if None, the full dataset is used.
        patience_tol : float, optional
            Tolerance for early stopping based on ELBO improvements.
        window_size : int, optional
            Number of epochs to wait before early stopping if no improvement is observed.
        """
        self.seed = seed
        self.n_epochs = n_epochs
        self.patience_tol = patience_tol  
        self.window_size = window_size
        self.S = S
        self.batch_size = batch_size
        self.model_interface = model_interface
        self.latent_vars_config = latent_variables
        self.rng_key = jax.random.PRNGKey(self.seed)

        self._process_full_rank_configs() #new pre processing step

        self.variational_dists_class = self._init_variational_dists_class()
        self.phi = self._init_phi()
        self.fixed_distribution_params = self._init_fixed_distribution_params()

        self.initial_distributions = self._validate_and_build_distributions()

        self.opt_state, self.optimizer = self._init_optimizer()
        self.elbo_values = []

        try:
            self.dim_data = next(
                var.value.shape[0] 
                for var in self.model_interface.model.vars.values() 
                if getattr(var, "observed", True)
            )
        except StopIteration:
            raise ValueError("No observed data found in model.")

    def _init_variational_dists_class(self):
        """Initialize variational distribution classes."""
        variational_dists_class = { 
            self._config_key(config): config["dist_class"]
            for config in self.latent_vars_config
        }
        return variational_dists_class

    def _init_phi(self):
        """Initialize the phi dictionary."""
        phi = {
            self._config_key(config): config["phi"]
            for config in self.latent_vars_config
        }
        return phi

    def _init_fixed_distribution_params(self):
        """Initialize fixed distribution parameters."""
        fixed_distribution_params = {
            self._config_key(config): config["fixed_distribution_params"] 
            if config["fixed_distribution_params"] is not None else {}  
            for config in self.latent_vars_config
        }
        return fixed_distribution_params

    def _init_transform_dict(self):
        """Initialize the transform dictionary."""
        optim_dict = {
            self._config_key(config): config["optimizer_chain"]
            for config in self.latent_vars_config
                }
        return optim_dict

    def _config_key(self, config):
        """Generate a configuration key from variable names."""
        return config["names"][0] if len(config["names"]) == 1 else config["full_rank_key"]

    def _build_distribution(self, dist_class, phi, fixed_distribution_params):
        """Builds a TFP distribution with given parameters phi and fixed_distribution_params."""
        return dist_class(**phi, **fixed_distribution_params)
    
    def _process_full_rank_configs(self):
        """
        Process configurations for Full-Rank latent variables and 
        checks dimensions for configurations with multiple variables and sets 
        a unique full_rank_key.
        """
        model_params = self.model_interface.get_params()

        for config in self.latent_vars_config:
            names = config["names"]

            if len(names) > 1:
                dims = []
                for pname in names:
                    if pname not in model_params:
                        raise KeyError(f"Parameter {pname} not found in model parameters")
                    dims.append(math.prod(model_params[pname].shape))

                total_dim = sum(dims)
                phi_conf = config["phi"]

                if phi_conf["loc"].shape[0] != total_dim:
                    raise ValueError(f"Dimension mismatch for full rank latent variables {names}: "
                                     f"expected loc dim {total_dim}, got {phi_conf['loc'].shape[0]}"
                                     )
                
                expected_len = total_dim * (total_dim + 1) // 2
                if phi_conf["log_cholesky_parametrization"].shape[0] != expected_len:
                    raise ValueError(
                        f"Dimension mismatch for full rank latent variables {names}: "
                        f"expected a flattened Cholesky of length {expected_len}, "
                        f"got {phi_conf['log_cholesky_parametrization'].shape[0]}"
                    )

                config["full_rank_key"] = "Full Rank:" + "_".join(names)

    def _validate_and_build_distributions(self):
        """Validate matching, consistency and build the variational distributions."""
        phi_keys = set(self.phi.keys())
        fixed_keys = set(self.fixed_distribution_params.keys())
        dist_keys = set(self.variational_dists_class.keys())

        if not (phi_keys == fixed_keys == dist_keys):
            raise ValueError(
                f"Mismatch in keys: phi_keys={phi_keys}, fixed_keys={fixed_keys}, dist_keys={dist_keys}"
            )

        distributions = {
            key: self._build_distribution(
                self.variational_dists_class[key],
                self.phi[key],
                self.fixed_distribution_params[key]
            )
            for key in phi_keys
        }
        return distributions

    def _init_optimizer(self):
        """Initialize the optimizer state and transformation."""
        def label_fn(params): 
            return {k: k for k in params if k in self.phi}

        optim_dict = self._init_transform_dict()
        tx = optax.multi_transform(optim_dict, label_fn)
        opt_state = tx.init(self.phi)
        return opt_state, tx


    def fit(self):
        """
        Run the optimization loop to update variational parameters by maximizing the negative ELBO.

        This method iterates over the specified number of epochs. At each epoch, the data is
        partitioned into batches (if a batch_size is provided) and, for each batch, a jitted 
        'step' function is executed to compute the ELBO, its gradients, and update the variational
        parameters. It also implements early stopping based on a patience threshold.
        """
        @partial(jax.jit, static_argnames=['batch_size', 'S'])
        def step(current_phi, opt_state, rng_key, dim_data, batch_size, batch_indices, S): 
            """
            Perform a single optimization step.

            Parameters
            ----------
            current_phi : dict
                Current variational parameters.
            opt_state : object
                Current optimizer state.
            rng_key : jax.random.PRNGKey
                Current JAX random key.
            dim_data : int
                Total number of data points.
            batch_size : int
                Size of the batch.
            batch_indices : array-like
                Indices for the current batch.
            S : int
                Number of samples used in the ELBO estimation.

            Returns
            -------
            new_phis : dict
                Updated variational parameters.
            new_opt_state : object
                Updated optimizer state.
            loss_val : float
                Computed loss value (negative ELBO) for the current batch.
            new_rng_key : jax.random.PRNGKey
                Updated random key.
            """
            (loss_val, new_rng_key), grads = jax.value_and_grad(
                lambda p, key: self._elbo(p, key, dim_data, batch_size, batch_indices, S), 
                has_aux=True
            )(current_phi, rng_key)

            updates, new_opt_state = self.optimizer.update(grads, opt_state, current_phi)
            new_phis = optax.apply_updates(current_phi, updates)
            return new_phis, new_opt_state, loss_val, new_rng_key
        
        phi = self.phi
        opt_state = self.opt_state
        rng_key = self.rng_key

        best_elbo = -float("inf")
        window_counter = 0
        early_stopping_enabled = (self.patience_tol is not None and self.window_size is not None)
        dim_data = self.dim_data 
        batch_size = self.batch_size

        number_batches = dim_data // batch_size if self.batch_size is not None else 1

        for epoch in range(self.n_epochs):
            rng_key, perm_key = jax.random.split(rng_key)
            all_indices = jax.random.permutation(perm_key, dim_data)
            batch_indices_list = jnp.array_split(all_indices, number_batches)
            
            epoch_elbos = []
            for batch_indices in batch_indices_list:
                phi, opt_state, loss_val, rng_key = step(
                    phi, opt_state, rng_key, dim_data, batch_size, batch_indices, self.S
                    )
                epoch_elbos.append(float(-loss_val))

            current_elbo = jnp.mean(jnp.array(epoch_elbos))
            self.elbo_values.append(float(current_elbo))

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}, ELBO: {current_elbo:.4f}")

            if early_stopping_enabled: 
                if current_elbo > best_elbo + self.patience_tol:
                    best_elbo = current_elbo
                    window_counter = 0
                else:
                    window_counter += 1

                if window_counter >= self.window_size:
                    print(f"Early stopping at epoch {epoch+1} with ELBO {current_elbo:.4f}")
                    break

        self.phi = phi
        self.opt_state = opt_state
        self.rng_key = rng_key
        self.final_variational_distributions = self.get_final_distributions()

    def _elbo(self, phi, rng_key, dim_data, batch_size, batch_indices, S):
        """
        Compute the negative ELBO (Evidence Lower Bound) for variational inference.

        Parameters
        ----------
        phi : dict
            Current variational parameters.
        rng_key : jax.random.PRNGKey
            Random key for sampling.
        dim_data : int
            Total number of data points.
        batch_size : int
            Batch size used for subsetting the data.
        batch_indices : array-like
            Indices corresponding to the current batch.
        S : int
            Number of Monte Carlo samples.

        Returns
        -------
        float
            Negative ELBO (loss) computed over S Monte Carlo samples.
        jax.random.PRNGKey
            Updated random key.
        """
        rng_key, subkey = jax.random.split(rng_key)
        num_samples = S
        subkeys = jax.random.split(subkey, num_samples)

        @jax.jit
        def _single_sample_elbo(rng_key_sample):
            """Compute the ELBO for a single sample by accessing the model via the Interface instance."""
            samples, log_det_jac, log_q = self._sample_variational(phi, rng_key_sample)
            log_prob = self.model_interface.compute_log_prob(samples, dim_data, batch_size, batch_indices)  
            return  (log_prob + log_det_jac - log_q) 

        elbo_samples = jax.vmap(_single_sample_elbo)(subkeys)
        elbo = jnp.mean(elbo_samples)
        return -elbo, rng_key
    
    def _apply_transform(self, z, transform_spec):
        """
        Apply a transformation to variable z and compute the log-determinant of its Jacobian.

        Parameters
        ----------
        z : jnp.ndarray
            Input variable to be transformed.
        transform_spec : callable or tfb.Bijector or None
            Transformation to apply.

        Returns
        -------
        tuple
            (z_transformed, ldj) where ldj is the log-determinant of the transformation.
        """
        if transform_spec is None:
            return z, 0.0

        elif callable(transform_spec) and not hasattr(transform_spec, "forward"):
            return transform_spec(z)

        elif hasattr(transform_spec, "forward") and hasattr(transform_spec, "forward_log_det_jacobian"):
            z_transformed = transform_spec.forward(z)
            event_ndims = 1 if z.ndim == 1 else 0
            ldj = transform_spec.forward_log_det_jacobian(z, event_ndims=event_ndims)
            if ldj.ndim > 0:
                ldj = jnp.sum(ldj)
            return z_transformed, ldj
        else:
            raise ValueError("Only tfb.Bijector instances and Python callables are supported as transforms")

    def _sample_single_variable(self, pname, phi, rng_key, transform_spec):
        """
        Sample a single latent variable using its fully reparameterizable 
        variational distribution.

        Parameters
        ----------
        pname : str
            Name of the latent variable.
        phi : dict
            Dictionary of variational parameters.
        rng_key : jax.random.PRNGKey
            Random key for sampling.
        transform_spec : callable or tfb.Bijector or None
            Transformation to apply to the sampled variable.

        Returns
        -------
        tuple
            (z_transformed, ldj, log_q, rng_key) where z_transformed is the sampled and transformed variable,
            ldj is the log-determinant of the Jacobian, log_q is the log probability under the variational distribution,
            and rng_key is the updated random key.
        """
        pval = phi[pname]
        
        dist_obj = self._build_distribution(
            self.variational_dists_class[pname],
            pval,
            self.fixed_distribution_params[pname]
        )

        if dist_obj.reparameterization_type == tfd.FULLY_REPARAMETERIZED: #rm
            rng_key, subkey = jax.random.split(rng_key)
            z = dist_obj.sample(seed=subkey)
            log_q = dist_obj.log_prob(z)
            z_transformed, ldj = self._apply_transform(z, transform_spec)
        else:
            raise NotImplementedError("Only fully reparameterized distributions are supported so far.")
        
        return z_transformed, ldj, log_q, rng_key
    

    def _sample_full_rank(self, config, phi, rng_key, name_to_transform): #doesnt need error anymore because of only using MultivariateNormalTriL
        """
        Sample latent variables jointly using a Full-Rank variational distribution a self created 
        instance of a TFP distribution: the MultivariateNormalLogCholeskyParametrization.

        Parameters
        ----------
        config : dict
            Configuration for the full-rank latent variable group.
        phi : dict
            Dictionary of variational parameters.
        rng_key : jax.random.PRNGKey
            Random key for sampling.
        name_to_transform : dict
            Mapping from variable names to their transformation specifications.

        Returns
        -------
        tuple
            (samples, total_ldj, log_q, rng_key) where samples is a dict of transformed samples,
            total_ldj is the sum of log-determinants, log_q is the log probability of the sample,
            and rng_key is the updated random key.
        """
        full_rank_key = config["full_rank_key"]
        pval = phi[full_rank_key]
        
        dist_obj = self._build_distribution(
            self.variational_dists_class[full_rank_key],
            pval,
            self.fixed_distribution_params[full_rank_key]
        )
        
        rng_key, subkey = jax.random.split(rng_key)
        z_full_rank = dist_obj.sample(seed=subkey)
        log_q = dist_obj.log_prob(z_full_rank)

        model_params = self.model_interface.get_params()
        dims = [math.prod(model_params[pname].shape) for pname in config["names"]]
        total_dim = sum(dims)
        z_full_rank_flat = jnp.ravel(z_full_rank)
        if z_full_rank_flat.shape[0] != total_dim:
            raise ValueError(
                f"Dimension mismatch for full rank latent variables {config['names']}: "
                f"expected {total_dim}, got {z_full_rank_flat.shape[0]}"
            )

        cum_dims = []
        running_sum = 0
        for d in dims[:-1]:
            running_sum += d
            cum_dims.append(running_sum)
        splits = jnp.split(z_full_rank_flat, cum_dims)

        samples = {}
        total_ldj = 0.0
        for i, pname in enumerate(config["names"]):
            expected_shape = model_params[pname].shape
            z_ind = jnp.reshape(splits[i], expected_shape)
            transform_spec = name_to_transform[pname]
            z_transformed, ldj = self._apply_transform(z_ind, transform_spec)
            samples[pname] = z_transformed
            total_ldj += ldj

        return samples, total_ldj, log_q, rng_key
    
    def _sample_variational(self, phi, rng_key):
        """
        Sample from the variational distribution for all latent variables.

        Parameters
        ----------
        phi : dict
            Dictionary of variational parameters.
        rng_key : jax.random.PRNGKey
            Random key for sampling.

        Returns
        -------
        tuple
            (samples, total_ldj, total_log_q) where samples is a dict of all 
            sampled and transformed latent variables, total_ldj is the cumulative 
            log-determinant, and total_log_q is the sum of log probabilities.
        """
        samples = {}
        total_ldj = 0.0
        total_log_q = 0.0

        name_to_transform = {
            pname: config.get("transform", None)
            for config in self.latent_vars_config
            for pname in config["names"]
        }

        for config in self.latent_vars_config:
            if len(config["names"]) == 1:
                pname = config["names"][0]
                z_transformed, ldj, log_q, rng_key = self._sample_single_variable(
                    pname, phi, rng_key, name_to_transform[pname]
                )
                samples[pname] = z_transformed
                total_ldj += ldj
                total_log_q += log_q
            else:
                
                full_samples, ldj, log_q, rng_key = self._sample_full_rank(
                    config, phi, rng_key, name_to_transform
                )
                samples.update(full_samples)
                total_ldj += ldj
                total_log_q += log_q

        return samples, total_ldj, total_log_q

    def get_final_distributions(self): 
        """
        Construct and return the final variational distributions after applying specified 
        transformations back into constrained space by applying bijectors.

        Returns
        -------
        dict
            Mapping from latent variable names to their final variational distribution objects
            while providing valid TFP distributions.
        """
        final_results = {}

        for config in self.latent_vars_config:
            names = config["names"]
            transform = config.get("transform", None)
            dist_class = self.variational_dists_class[self._config_key(config)]
            phi_original = self.phi[self._config_key(config)]

            phi_transformed = {}
            for param_name, param_value in phi_original.items():
                if transform is None:
                    phi_transformed[param_name] = param_value  
                elif callable(transform) and not hasattr(transform, "forward"):
                    phi_transformed[param_name], _ = transform(param_value)  
                elif hasattr(transform, "forward"):
                    phi_transformed[param_name] = transform.forward(param_value)  
            
            final_distribution = self._build_distribution(
                dist_class, phi_transformed, self.fixed_distribution_params[self._config_key(config)]
            )
            final_results.update({name: final_distribution for name in names})

        return final_results

    def plot_elbo(self, title="ELBO Progress", xlabel="Iterations", ylabel="Negative ELBO", style="whitegrid", color="blue", save_path=None):
        """
        Plot the ELBO progress over iterations.

        Parameters
        ----------
        title : str, optional
            Title of the plot.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        style : str, optional
            Seaborn style for the plot.
        color : str, optional
            Color for the line plot.
        save_path : str, optional
            If provided, the plot will be saved to the specified path.
        """
        sns.set_theme(style=style)
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(self.elbo_values)), y=self.elbo_values, color=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim()
        if save_path:
            plt.savefig(save_path)
        plt.show()

