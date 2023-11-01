# Project description

This project contains tools and scripts used to analyze and interpret seismic data for first break
picking. These have been developed by Mila in collaboration with the Geological Survey of Canada
(GSC), which is part of National Resources Canada (NRCan). We also provide links below to our
multi-survey seismic dataset which is generously hosted by NRCan.

Two publications are associated with this repository:

"Deep Learning Benchmark for First Break Detection from Hardrock Seismic Reflection Data",
St-Charles et al., GEOPHYSICS, 2023: [[Open Access link]](https://doi.org/10.1190/geo2022-0741.1)
```
@article{stcharles2023hardpicks_preprint,
  title={Deep Learning Benchmark for First Break Detection from Hardrock Seismic Reflection Data},
  author={St-Charles, Pierre-Luc and Rousseau, Bruno and Ghosn, Joumana and Bellefleur, Gilles and Schetselaar, Ernst},
  journal={Geophysics},
  volume={89},
  number={1},
  pages={1--68},
  year={2023},
  publisher={Society of Exploration Geophysicists}
}
```

"A Multi-Survey Dataset and Benchmark for First Break Picking in Hard Rock Seismic Exploration",
St-Charles et al., ML4PS 2021 (Neurips 2021 Workshop): [[PDF link]](https://ml4physicalsciences.github.io/2021/files/NeurIPS_ML4PS_2021_3.pdf)
```
@inproceedings{stcharles2021hardpicks_workshop,
  title={A multi-survey dataset and benchmark for first break picking in hard rock seismic exploration},
  author={St-Charles, Pierre-Luc and Rousseau, Bruno and Ghosn, Joumana and Bellefleur, Gilles and Schetselaar, Ernst},
  booktitle={Proc. of the 2021 NeurIPS Workshop on Machine Learning for the Physical Sciences (ML4PS)},
  year={2021}
}
```

# Hardpicks Dataset

Before downloading any data, make sure you read and understand the data licensing terms below.

### Brunswick and Halfmile Lake 3D Surveys License

Mila and Natural Resources Canada have obtained licences from Glencore Canada Corporation and Trevali
Mining Corporation to distribute field seismic data from the Brunswick 3D and Halfmile Lake 3D seismic
surveys, respectively, under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)](
https://creativecommons.org/licenses/by/4.0/). These datasets are in the Hierarchical Data Format
(HDF5) and have first arrival labels included in trace headers.

### Lalor and Sudbury 3D Surveys License

The Lalor 3D and Sudbury 3D seismic data are distributed under the [Open Government Licence – Canada]( https://open.canada.ca/en/open-government-licence-canada). Canada grants to the licensee a non-exclusive,
fully paid, royalty-free right and licence to exercise all intellectual property rights in the data. This
includes the right to use, incorporate, sublicense (with further right of sublicensing), modify, improve,
further develop, and distribute the Data; and to manufacture or distribute derivative products.

The formatting of these datasets is similar to the other two.

Please use the following attribution statement wherever applicable:

    Contains information licensed under the Open Government Licence – Canada.

### Download links

The HDF5 files are hosted on AWS, and can be downloaded directly:
 - [Brunswick](https://d3sakqnghgsk6x.cloudfront.net/Brunswick_3D/Brunswick_orig_1500ms_V2.hdf5.xz)
 - [Halfmile Lake](https://d3sakqnghgsk6x.cloudfront.net/Halfmile_3D/Halfmile3D_add_geom_sorted.hdf5.xz)
 - [Lalor](https://d3sakqnghgsk6x.cloudfront.net/Lalor_3D/Lalor_raw_z_1500ms_norp_geom_v3.hdf5.xz)
 - [Sudbury](https://d3sakqnghgsk6x.cloudfront.net/Sudbury_3D/preprocessed_Sudbury3D.hdf.xz)

### Data loading demo

We demonstrate how to parse and display the raw data in [this notebook](./examples/local/fbp_data_loading_demo.ipynb).

### Cross-validation folds

The cross-validation folds used in the NeurIPS 2021 ML4PS workshop paper are as follow:
```
Fold Sudbury:
Train: Halfmile, Lalor;
Valid: Brunswick;
Test: Sudbury

Fold Brunswick:
Train: Sudbury, Halfmile;
Valid: Lalor;
Test: Brunswick

Fold Halfmile:
Train: Lalor, Brunswick;
Valid: Sudbury;
Test: Halfmile

Fold Lalor:
Train: Brunswick, Sudbury;
Valid: Halfmile;
Test: Lalor
```

For the GEOPHYSICS 2023 version, refer to Table 3 of the [paper](https://doi.org/10.1190/geo2022-0741.1).

### Acknowledgements

We thanks Glencore Canada Corporation and Trevali Mining Corporation for providing access and allowing us
to include and distribute the Brunswick 3D and Halfmile 3D seismic data as part of this benchmark dataset.
We also thank E. Adam, S. Cheraghi, and A. Malehmir for providing first breaks for the Brunswick, Halfmile,
and Sudbury data.


# Hardpicks package description

The `hardpicks` package is provided as a reference for researchers to see how we implemented and
trained the models used in the experiments described in our papers. It is NOT a production-ready
codebase, and it requires a decent understanding of Python and PyTorch to dig into and use. 

For in-depth examples on how to parse the proposed dataset, on how to use different parts of
the `hardpicks` package, and for information about hyperparameters used in configuration files,
refer to the notebooks present in the `examples` subdirectory [here](./examples/local).

With the proposed `hardpicks` package and its API, the entry point to train deep neural networks
is the following script:

    hardpicks/main.py

This script takes in many command-line arguments to describe the model that should be trained and
where the data is located. Most of the required arguments come under the form of YAML configuration
files; some examples of these can be found in the `examples` subdirectory [here](./examples/local).

The following tables describe at a high level the content of the code base.

| Folder       | Description                                                                                                                                                                          |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `config/`    | Configuration files and utilities for experiment definitions and project management.                                                                                                 |
| `data/`      | Contains the directory structure where raw data will be parsed as well as other task-specific data files (e.g. bad gather lists, fold configurations, mlflow analysis scripts, ...). |
| `docs/`      | Scripts used to generate HTML documentation for the project.                                                                                                                         |
| `examples/`  | Many examples on how to execute the code on different platforms.                                                                                                                     |
| `hardpicks/` | Python package that contains all the modules and utilities to perform deep learning model training and evaluation.                                                                   |
| `tests/`     | Battery of unit tests.                                                                                                                                                               |

Some utility scripts in the code base may be of particular interest to new users:

| Script                                                     | Description                                                                                                                                                        |
|------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `hardpicks/analysis/fbp/bad_gathers/dataframe_analyzer.py` | GUI inspection tool used to identify poorly annotated line gathers for first break picking.                                                                        |
| `./linting_test.sh`                                        | Used in our Continuous Integration (CI) pipeline to insure code quality. Not directly related to the project.                                                      |

All functionalities that are related to the training or evaluation of predictive models are
implemented as part of the `hardpicks` package. A brief description of its subpackages is provided
below. For more information on these, visit the package's [README pages](hardpicks/README.md).

| Library Subpackages   | Description                                                                                                                                                                                                                                                                                            |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `hardpicks.analysis`  | Contains standalone scripts and utilities used for the generation of plots and tables used for data analysis. Some of these scripts may be outdated, as they are typically not updated each time datasets and preprocessing techniques change. More information [here](hardpicks/analysis/README.md).  |
| `hardpicks.data`      | Contains classes and functions used for data loading. This subpackage is used by both analysis scripts and model training/evaluation scripts.  More information [here](hardpicks/data/README.md).                                                                                                      |
| `hardpicks.metrics`   | Contains evaluation utilities used to compute metrics and produce reports during/after model training.  More information [here](hardpicks/metrics/README.md).                                                                                                                                          |
| `hardpicks.models`    | Contains modules and layer implementations used to construct predictive models as well as optimizer and scheduler implementations used for training.  More information [here](hardpicks/models/README.md).                                                                                             |
| `hardpicks.utils`     | Contains generic utility functions used across all other subpackages. More information [here](hardpicks/utils/README.md).                                                                                                                                                                              |

### License for the software package

Copyright (c) 2023 Mila - Institut Québecois d'Intelligence Artificielle.

This software package is licensed under Apache 2.0 terms. See the [LICENSE](./LICENSE) file for
more information. For the license acknowledgements of 3rd-party dependencies, refer to the
[ACKNOWLEDGEMENTS](./ACKNOWLEDGEMENTS) file.

### Instructions to install the software

This project relies on Conda to create a virtual environment and manage dependencies. See
https://anaconda.org/ for details.

Create the conda environment (this might take some time, as most packages are pinned!):

    conda env create -f environment.yml

Activate the environment:

    conda activate hardpicks-dev

Install the project for development:

    python setup.py develop

Note that this command indicates that `main.py` is the entrypoint, and correspondingly this script
is in the executable path after installation.

### Executing the tests

The code base is unit tested by using the `pytest` library. To run the tests (from the root folder):

    pytest

This does not require the presence of GPUs.

### Executing a single job

Note that the code should already be installed at this point. The `main.py` script can be invoked
as follows:

    main  --data $DATA_BASE_DIR  \            # location of the data
          --output $OUTPUT \                  # output directory
          --mlflow-output=$MLFLOW \           # where to direct mlflow logs (OPTIONAL)
          --tensorboard-output=$TENSORBOARD \ # where to direct tensorboard logs (OPTIONAL)
          --config $CONFIG \                  # path to model and training configuration parameters
          --gpu "0" \                         # which GPU to train on (relevant when many jobs run on a multi-GPU machine)
          --disable-progressbar >& $CONFIG_DIR/$LOG_FILENAME &

Note that code execution is in principle possible on a CPU, but is extremely slow for serious 
training. 

### Executing hyperparameter searches with Orion 

Running a hyperparameter search with Orion can be done as follows.

    orion -v hunt --config $ORION_CONFIG main  \
                  --data $DATA_BASE_DIR  \
                  --output $OUTPUT/'{exp.working_dir}/{exp.name}_{trial.id}/' \
                  --mlflow-output=$MLFLOW \
                  --tensorboard-output=$TENSORBOARD \
                  --config $CONFIG \
                  --gpu "0" \
                  --disable-progressbar >& $CONFIG_DIR/$LOG_FILENAME &

For a concrete example of the Orion config file, see `data/fbp/folds/orion_config.yaml`. Also, the
model and training configuration file should indicate which parameters Orion should search over.
For a concrete example, see `data/fbp/folds/foldA.yaml`.

### Building the documentation

To automatically generate the documentation for this project, cd to the `docs` folder then run:

    make html

To view the documents locally, open `docs/_build/html/index.html` in a browser.
