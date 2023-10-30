from torch import nn


class Residual(nn.Module):
    """Residual connection with constant affine weights. Can simulate standard residual, no residual, and "constant gates"."""

    def __init__(self, i_layer, d_input, d_model, alpha=1.0, beta=1.0):
        # print("ConstantResidual extra kwargs", kwargs)
        super().__init__()
        assert (d_input == d_model) or alpha == 0.0
        self.i_layer = i_layer
        self.d_input = d_input
        self.d_model = d_model
        self.alpha = alpha
        self.beta = beta

    @property
    def d_output(self):
        return self.d_model

    def forward(self, x, y, transposed):  # TODO documentation of transposed
        y = self.beta * y if self.beta != 1.0 else y
        return self.alpha * x + y if self.alpha else y
