### `hardpicks.models` subpackage

This subpackage contains classes that implement different types of blocks, layers, and entire models
that can be used to tackle different tasks. Most models share the same [base interface](./base.py)
that is derived from the PyTorch-Lightning base interface (`pl.LightningModule`), meaning they are
meant to be used as part of PyTorch-Lightning training/prediction experiments, which are launched
using the `hardpicks/main.py` script.

Models are automatically instantiated and used as part of our training and evaluation pipelines,
but they can also be created and used manually. See the [this module](./model_loader.py) for more
information.

Like for the [`data`](../data/README.md) subpackage, task-specific details are split into separate
folders; see `fbp` for first break picking.
