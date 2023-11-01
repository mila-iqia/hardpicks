import os
import typing


def get_data_dump_path(
    model_file_path: str,
    dataloader_type: str,
    data_dump_output_directory: str,
) -> typing.AnyStr:
    """Assembles and returns the path and file name for the data dump of prediction results.

    Args:
        model_file_path: path to the trained model that is used generate predictions.
        data_dump_output_directory: directory where the output data dump file should be written
        dataloader_type: type of dataloader that the model is applied to (one of train, valid, test)

    Returns:
        data_dump_file_path : path to where the output file will be written.
    """
    if model_file_path == '':
        # If the input string is empty, then there is no best trained model, which indicates
        # that the training phase was skipped.
        model_root_name = "no_training"
    else:
        model_filename = os.path.split(model_file_path)[1]
        model_root_name = os.path.splitext(model_filename)[0]
    data_dump_file_path = f"{data_dump_output_directory}/output_{model_root_name}_{dataloader_type}.pkl"
    return data_dump_file_path


def get_eval_prefix_based_on_data_dump_path(
    path: typing.AnyStr,
) -> typing.Optional[typing.AnyStr]:
    """If the provided path is associated with a train/valid/test set prefix, returns it.

    This function is to be used in conjunction with the `get_data_dump_path` defined in the same
    module. It is useful to figure out the origin of a dataset when generating its predictions
    inside a model, and to therefore give a more meaningful prefix to prediction results.
    """
    if path.endswith("_train.pkl"):
        return "train"
    elif path.endswith("_valid.pkl"):
        return "valid"
    elif path.endswith("_test.pkl"):
        return "test"
    return None
