This folder contains notebooks and configuration files that can be used to train and evaluate
models directly on a local machine. For example, to train a model to predict first break picks
from scratch and see how well it performs on a validation set, you can simply execute the
following command:

    python <PATH_TO_THE_PACKAGE_ROOT>/hardpicks/main.py \
        --data <PATH_TO_THE_DIRECTORY_WHERE_THE_RAW_DATA_IS_LOCATED>  \
        --output <PATH_TO_THE_DIRECTORY_WHERE_THE_OUTPUT_SHOULD_BE_SAVED> \
        --config <PATH_TO_THIS_FOLDER>/fbp-unet-mini.yaml

More concretely, if using the 'default' paths + symlinks and running from the project root:

    python ./hardpicks/main.py \
        --data ./data/fbp/data  \
        --output ./data/fbp/results \
        --config ./examples/local/fbp-unet-mini.yaml

For a more in-depth look at the execution of the training/evaluation process, see the notebooks
that are also provided in this directory. Specifically:
 - [`fbp_data_loading_demo.ipynb`](./fbp_data_loading_demo.ipynb): shows how to read and use the
   raw seismic data along with first break picks without using the rest of the package/API.
 - [`fbp_train_with_api.ipynb`](./fbp_train_with_api.ipynb): shows how to train a model to predict
   first break picks using the `hardpicks` package and API through the above `main.py` script.
 - [`fbp_predict_with_api.ipynb`](./fbp_predict_with_api.ipynb): shows how to evaluate a model to
   make it predict first break picks using the `hardpicks` package and API through the
   above `main.py` script.
