import typing


def get_value_or_default(
    input_value: typing.Optional[typing.Any],
    default_value: typing.Any,
    assert_val_type: bool = True,
) -> typing.Any:
    """Checks whether the input value is `None`, and if so, returns the default."""
    assert default_value is not None
    if input_value is None:
        return default_value
    if assert_val_type:
        assert isinstance(input_value, type(default_value)), \
            f"invalid input data type (is '{type(input_value)}', should be '{type(default_value)}')"
    return input_value
