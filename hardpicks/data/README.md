### `hardpicks.data` subpackage

This subpackage contains classes used to parse data from files on disk, clean it, and preprocess it.
The design that was adopted for the classes can be described as follows: a data "parser" will load
the data from disk, this parser can be "wrapped" by other classes to transform the loaded data
before it is delivered downstream, and a data "loader" (based on PyTorch's terminology) is used
downstream to package multiple examples provided by the parser into a minibatch (for training or
evaluation).

For first break picking, all relevant classes are located in the `fbp` subdirectory.

Below is a list of data parser modules and class names, as well as a brief description of what they
do. For more information on each parser, refer to the docstrings that accompany the class
definition in each module.

| Module                                |  Class                       | Description                                                                                                                                                                                                                            |
|---------------------------------------|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `...data.fbp.gather_parser`           | `ShotLineGatherDataset`      | Loads instances of 'shot line gathers', namely the intersection of traces for a single receiver line and a single shot, based on HDF5 archives of raw trace data.                                                                      |

Most modules in this subpackage contain class definitions, but some of them also have "entrypoints"
that can be used to quickly test whether data can be loaded or not. These show up at the bottom
of each module following the `if __name__ == '__main__':` line. By default, they will likely attempt
to load data using the root paths specified in the
[`__init__.py` module of the package root.](../__init__.py).

#### Data preprocessing

Note that some of the raw data may need to be preprocessed (read from disk, cleaned up, transformed,
repackaged) in order to speed up analysis and training.

For first break picking data, this preprocessing applies to all datasets, and it can be completed by
running [this script](../first_break_picking_preprocess.py) as follows:

    $ python "<PATH_TO_PROJECT_DIR>/hardpicks/first_break_picking_preprocess.py"

The data is expected to be located within the `DATA_ROOT_DIR` subdirectory structure
defined in [the root package's init module](../__init__.py).
