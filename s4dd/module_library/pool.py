import torch.nn.functional as F
from einops import rearrange, reduce

from .sequence_module import SequenceModule
from .util_modules import LinearActivation


class DownAvgPool(SequenceModule):
    def __init__(self, d_input, stride=1, expand=None, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

        if self.expand is not None:
            self.linear = LinearActivation(
                d_input,
                d_input * expand,
                transposed=transposed,
            )

    def forward(self, x):
        if not self.transposed:
            x = rearrange(x, "b ... d -> b d ...")

        if self.stride > 1:
            # einops appears slower than F
            if x.ndim == 3:
                x = F.avg_pool1d(x, self.stride, self.stride)
            elif x.ndim == 4:
                x = F.avg_pool2d(x, self.stride, self.stride)
            else:
                # Reduction string e.g. "b d (l1 2) (l2 2) -> b d l1 l2"
                reduce_str = (
                    "b d "
                    + " ".join([f"(l{i} {self.stride})" for i in range(x.ndim - 2)])
                    + " -> b d "
                    + " ".join([f"l{i}" for i in range(x.ndim - 2)])
                )
                x = reduce(x, reduce_str, "mean")

        # if self.expand > 1:
        #     x = repeat(x, 'b d ... -> b (d e) ...', e=self.expand)

        if not self.transposed:
            x = rearrange(x, "b d ... -> b ... d")
        if self.expand is not None:
            x = self.linear(x)
        return x, None

    def step(self, x, state, **kwargs):
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        if self.expand is None:
            return self.d_input
        else:
            return self.d_input * self.expand
