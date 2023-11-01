def get_number_of_elements_in_generator(generator):
    """Get number of elements in a generator.

    Count the number of objects in the generator without holding those blocks in memory.
    We of course assumes this is not infinite.

    Args:
        generator (generator): a generator

    Returns:
        number of objects in generator
    """
    return sum(1 for _ in generator)
