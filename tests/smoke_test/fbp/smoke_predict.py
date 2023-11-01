import mock

from hardpicks import predict
from tests.test_smoke_main import fake_get_site_info_array

if __name__ == '__main__':
    mock_target = (
        "hardpicks.data.fbp.site_info.get_site_info_array"
    )
    with mock.patch(mock_target, new=fake_get_site_info_array):
        predict.predict()
