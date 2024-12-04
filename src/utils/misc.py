import torch
import numpy as np

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from matplotlib.figure import Figure
from os import devnull


# silence output to stdout and stderr
# https://stackoverflow.com/a/52442331/2426888
@contextmanager
def hush():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def norm(batch: torch.Tensor) -> torch.Tensor:
    assert batch.dim() == 2, f"expected a 2D tensor (a batch of vectors), but got a {batch.dim()}D one"
    batch_norm = batch.norm(p=2, dim=-1).unsqueeze(-1)
    return batch / batch_norm, batch_norm


def is_integer(batch) -> bool:
    return ((batch == 0) | (batch == 1)).all()


def fig_to_rgb_tensor(fig: Figure):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img_np = data.reshape((int(h), int(w), -1)).copy()
    # matplotlib puts channels last, pytorch puts channels first
    img_pt = torch.FloatTensor(img_np).permute(2, 0, 1)
    return img_pt
