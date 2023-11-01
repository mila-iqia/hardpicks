from pathlib import Path

import mock

from hardpicks.main import main, setup_arg_parser
from tests.fake_fbp_data_utilities import (
    create_fake_site_data,
    create_fake_hdf5_dataset,
)
from tests.smoke_test.fbp.smoke_test_data import train_specs1, train_specs2, train_specs3, valid_specs, test_specs, \
    fake_get_site_info_array


def create_fake_data(data_directory):
    data_directory_path = Path(data_directory)
    if not data_directory_path.exists():
        data_directory_path.mkdir(exist_ok=True)
        for specs in [train_specs1, train_specs2, train_specs3, valid_specs, test_specs]:
            hdf5_path = data_directory_path.joinpath(f"{specs.site_name}.hdf5")
            fake_data = create_fake_site_data(specs)
            create_fake_hdf5_dataset(fake_data, hdf5_path)


if __name__ == '__main__':
    parser = setup_arg_parser()
    args = parser.parse_args()
    create_fake_data(args.data)

    mock_target = (
        "hardpicks.data.fbp.site_info.get_site_info_array"
    )
    with mock.patch(mock_target, new=fake_get_site_info_array):
        main()
