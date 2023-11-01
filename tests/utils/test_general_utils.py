import pytest
import numpy as np

from hardpicks.utils.general_utils import (
    get_number_of_elements_in_generator,
)


def fake_generator(number_of_elements):
    counter = 0
    # One by one yield next Fibonacci Number
    while counter < number_of_elements:
        yield np.random.randint(100)
        counter += 1


@pytest.mark.parametrize("number_of_elements", [2, 4, 8])
def test_get_number_of_elements_in_generator(number_of_elements):

    gen = fake_generator(number_of_elements)
    assert number_of_elements == get_number_of_elements_in_generator(gen)
