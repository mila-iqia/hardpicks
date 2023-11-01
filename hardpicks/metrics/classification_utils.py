import numpy as np
from einops import rearrange, parse_shape

CANNOT_PREDICT_INDICATOR = -9999  # careful not to overflow np.int16!


def get_class_index_from_probabilities(probabilities: np.array, prediction_threshold: float = None) -> np.array:
    """Get class index from probabilities.

    This method extracts the index of the highest probability class. If an optional threshold is provided,
    only a predicted class probability that exceeds this threshold will be considered: otherwise, the prediction
    will be the CANNOT_PREDICT_INDICATOR integer. We don't want to use "-1" or DONTCARE_SEGM_LABEL here
    as it might lead to subtle bugs since "-1" can be a valid index.

    Args:
        probabilities (np.array): an array assumed to be of shape [C, H, W, ...], where C is the number of
            classes. The first dimension is assumed to contain probabilities that should sum to 1.
        prediction_threshold (optional: float): value below which a prediction cannot be returned; the
            CANNOT_PREDICT_INDICATOR will be used to identify these cases.

    Returns:
        classes (np.array): an array of shape [H, W, ...] where each entry is the most likely class, or
        CANNOT_PREDICT_INDICATOR if all probabilities are below the threshold.
    """
    # Build up a "shape string" to tell einops about all the spatial dimensions
    spatial_dimensions = len(probabilities.shape) - 1
    shape_string = ""
    for i in range(spatial_dimensions):
        shape_string += f" d{i}"

    shape_dict = parse_shape(probabilities, f'_ {shape_string}')
    flat_probabilities = rearrange(probabilities, "c ... -> c (...)")
    flat_classes = np.argmax(flat_probabilities, axis=0).astype(np.int16)

    if prediction_threshold is not None:
        assert 0. <= prediction_threshold <= 1., \
            "the prediction threshold should be NONE or a number between 0 and 1"

        maximum_probabilities = np.max(flat_probabilities, axis=0)

        flat_classes[maximum_probabilities < prediction_threshold] = CANNOT_PREDICT_INDICATOR

    classes = rearrange(flat_classes, f"({shape_string})-> {shape_string}", **shape_dict)

    return classes
