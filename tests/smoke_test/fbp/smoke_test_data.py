import os

from tests.fake_fbp_data_utilities import FakeSiteSpecifications, IntegerRange

number_of_time_samples = 64
receiver_range = IntegerRange(min=32, max=64)

train_specs1 = FakeSiteSpecifications(
    site_name="train1",
    random_seed=13,
    samp_rate=4000,
    receiver_id_digit_count=4,
    first_break_field_name='SPARE3',
    number_of_time_samples=number_of_time_samples,
    number_of_shots=20,
    number_of_lines=30,
    shot_id_is_corrupted=False,
    range_of_receivers_per_line=receiver_range,
)

train_specs2 = FakeSiteSpecifications(
    site_name="train2",
    random_seed=41,
    samp_rate=2000,
    receiver_id_digit_count=6,
    first_break_field_name='SPARE2',
    number_of_time_samples=number_of_time_samples,
    number_of_shots=5,
    number_of_lines=7,
    shot_id_is_corrupted=False,
    range_of_receivers_per_line=IntegerRange(min=20, max=50),
)

train_specs3 = FakeSiteSpecifications(
    site_name="train3",
    random_seed=12341,
    samp_rate=1000,
    receiver_id_digit_count=4,
    first_break_field_name='SPARE2',
    number_of_time_samples=number_of_time_samples,
    number_of_shots=2,
    number_of_lines=17,
    shot_id_is_corrupted=False,
    range_of_receivers_per_line=IntegerRange(min=30, max=65),
)


valid_specs = FakeSiteSpecifications(
    site_name="valid",
    random_seed=141,
    samp_rate=1000,
    receiver_id_digit_count=4,
    first_break_field_name='SPARE1',
    number_of_time_samples=number_of_time_samples,
    number_of_shots=8,
    number_of_lines=4,
    shot_id_is_corrupted=False,
    range_of_receivers_per_line=IntegerRange(min=20, max=50),
)

test_specs = FakeSiteSpecifications(
    site_name="test",
    random_seed=666,
    samp_rate=4000,
    receiver_id_digit_count=6,
    first_break_field_name='SPARE4',
    number_of_time_samples=number_of_time_samples,
    number_of_shots=5,
    number_of_lines=9,
    shot_id_is_corrupted=False,
    range_of_receivers_per_line=IntegerRange(min=20, max=50),
)


def get_site_info(specs, data_dir: str):
    hdf5_path = os.path.join(data_dir, f"{specs.site_name}.hdf5")

    site_info = dict(
        site_name=specs.site_name,
        raw_hdf5_path=hdf5_path,
        processed_hdf5_path=hdf5_path,
        first_break_field_name=specs.first_break_field_name,
        raw_md5_checksum="00000000000",
        processed_md5_checksum="000000000000",
        receiver_id_digit_count=specs.receiver_id_digit_count,
    )
    return site_info


def fake_get_site_info_array(data_dir):
    site_info_array = [
        get_site_info(train_specs1, data_dir),
        get_site_info(train_specs2, data_dir),
        get_site_info(train_specs3, data_dir),
        get_site_info(valid_specs, data_dir),
        get_site_info(test_specs, data_dir),
    ]

    return site_info_array
