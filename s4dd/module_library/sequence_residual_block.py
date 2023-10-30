from torch import nn

from functools import partial

from .util_modules import Normalization, StochasticDepth, DropoutNd
from .sequence_module import SequenceModule
from .s4 import S4
from .ff import FF
from .pool import DownAvgPool
from .residual import Residual


class SequenceResidualBlock(SequenceModule):
    def __init__(
        self,
        d_input,
        i_layer=None,  # Only needs to be passed into certain residuals like Decay
        prenorm=True,
        dropout=0.0,
        tie_dropout=False,
        transposed=False,
        layer_config=None,  # Config for black box module
        residual=None,  # Config for residual function
        norm=None,  # Config for normalization layer
        pool=None,
        drop_path=0.0,
    ):
        super().__init__()

        self.i_layer = i_layer
        self.d_input = d_input
        # self.layer = instantiate(registry.layer, layer, d_input)
        # layer_config = layer.copy()
        # layer_cls = registry.get_layer(layer["_name_"])
        layer_config = layer_config.copy()
        if layer_config["_name_"] == "s4":
            layer_cls = S4
        elif layer_config["_name_"] == "ff":
            layer_cls = FF
        layer_config.pop("_name_")
        self.layer = layer_cls(d_input, **layer_config)

        self.prenorm = prenorm
        self.transposed = transposed

        # Residual
        # d_residual is the output dimension after residual
        if residual is None:
            self.residual = None
            self.d_residual = self.layer.d_output
        else:
            # self.residual = instantiate(
            #     residual_registry, residual, i_layer, d_input, self.layer.d_output
            # )
            self.residual = Residual(i_layer, d_input, self.layer.d_output)
            # instantiate(
            #     residual_registry, residual, i_layer, d_input, self.layer.d_output
            # )
            self.d_residual = self.residual.d_output

        # Normalization
        d_norm = d_input if self.prenorm else self.d_residual
        # We don't use config to directly instantiate since Normalization has some special cases
        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            self.norm = Normalization(d_norm, transposed=self.transposed, _name_=norm)
        else:
            self.norm = Normalization(d_norm, transposed=self.transposed, **norm)

        # Pool
        if pool is not None:
            self.pool = DownAvgPool(self.d_residual, transposed=self.transposed)

        # Dropout
        dropout_cls = (
            partial(DropoutNd, transposed=self.transposed)
            if tie_dropout
            else nn.Dropout
        )
        self.drop = dropout_cls(dropout) if dropout > 0.0 else nn.Identity()

        # Stochastic depth
        self.drop_path = (
            StochasticDepth(drop_path, mode="row") if drop_path > 0.0 else nn.Identity()
        )

    @property
    def d_output(self):
        return self.pool.d_output if self.pool is not None else self.d_residual

    @property
    def d_state(self):
        return self.layer.d_state

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def forward(self, x, state=None, **kwargs):
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm:
            y = self.norm(y)

        # Black box layer
        y, state = self.layer(y, state=state, **kwargs)

        # Residual
        if self.residual is not None:
            y = self.residual(x, self.drop_path(self.drop(y)), self.transposed)
        # Post-norm
        if self.norm is not None and not self.prenorm:
            y = self.norm(y)

        # Pool
        if self.pool is not None:
            y, _ = self.pool(y)

        return y, state

    def step(self, x, state, **kwargs):
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm:
            y = self.norm.step(y)

        # Black box layer
        y, state = self.layer.step(y, state, **kwargs)
        # Residual
        if self.residual is not None:
            y = self.residual(
                x, y, transposed=self.transposed
            )  # NOTE this would not work with concat residual function (catformer)
        # Post-norm
        if self.norm is not None and not self.prenorm:
            y = self.norm.step(y)

        # Pool
        if self.pool is not None:
            y, _ = self.pool(y)

        return y, state
