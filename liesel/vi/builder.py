from typing import Dict, List, Optional, Union
from tensorflow_probability.substrates import jax as tfp
from .interface import LieselInterface
from .optimizer import Optimizer

from typing import Callable
import tensorflow_probability.substrates.jax.bijectors as tfb

tfd = tfp.distributions


class OptimizerBuilder:
    def __init__(self, seed: int = 0, n_epochs: int = 10_000, lr: float = 1e-2):
        self.seed = seed
        self.n_epochs = n_epochs
        self.lr = lr
        self._model_interface: Optional[LieselInterface] = None
        self.latent_variables = []

    def set_model(self, interface: LieselInterface):
        self._model_interface = interface

    # def add_latent_variable(
    #     self,
    #     names: List[str],
    #     distribution: tfd.Distribution,
    #     transform: Optional[Union[str, Dict]] = None,
    #     optimizer: str = "adam"
    # ):
    #     self.latent_variables.append({
    #         "names": names,
    #         "distribution": distribution,
    #         "transform": transform,
    #         "optimizer": optimizer
    #     })
    

    def add_latent_variable(
        self,
        names: List[str],
        distribution: tfd.Distribution,
        transform: Optional[Union[Callable, tfb.Bijector]] = None,
        optimizer: str = "adam"
        ):
        """
        'transform' can be:
        - None (meaning no transform),
        - a Python callable, e.g.:
            def my_transform(z):
                z_transformed = jnp.exp(z)
                logdet = jnp.sum(z)
                return z_transformed, logdet
        - a TFP Bijector instance, e.g. tfb.Exp() or tfb.Sigmoid(low=a, high=b)

        Examples:
        builder.add_latent_variable(
            names=["theta"],
            distribution=tfd.Normal(0., 1.),
            transform=None
        )
        builder.add_latent_variable(
            names=["theta"],
            distribution=tfd.Normal(0., 1.),
            transform=tfb.Exp()
        )
        builder.add_latent_variable(
            names=["theta"],
            distribution=tfd.Normal(0., 1.),
            transform=lambda z: (jnp.exp(z), jnp.sum(z))
        )
        """
        self.latent_variables.append({
            "names": names,
            "distribution": distribution,
            "transform": transform,
            "optimizer": optimizer
        })


    def set_duration(self, n_epochs: int):
        self.n_epochs = n_epochs

    def build(self) -> "Optimizer":
        if self._model_interface is None:
            raise ValueError("Model interface not set. Call builder.set_model(...) first.")

        from .optimizer import Optimizer  
        return Optimizer(
            seed=self.seed,
            n_epochs=self.n_epochs,
            lr=self.lr,
            model_interface=self._model_interface,
            latent_variables=self.latent_variables
        )






















