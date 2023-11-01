### `hardpicks.metrics` subpackage

This subpackage contains classes that implement different types of evaluators used to measure the
success of predictive models on different tasks. All evaluators share the same
[base interface](./base.py) that ingests loaded data batches along with model predictions, and that
produces evaluation summaries.

Evaluators are automatically instantiated and used as part of our training and evaluation pipelines,
but they can also be created and used manually. See [this module](./eval_loader.py) for more
information.

Like for the [`data`](../data/README.md) subpackage, task-specific details are split into separate
folders; see `fbp` for first break picking.
