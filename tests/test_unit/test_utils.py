from unittest.mock import MagicMock, mock_open

from taskframe.client import Client


def mock_open_func(filename, *args, **kwargs):
    return str(filename)


custom_mock_open = mock_open()
custom_mock_open.side_effect = mock_open_func


def mock_client():
    client = Client()
    client.session = MagicMock()
    client.session.post.return_value.status_code = 200
    client.session.get.return_value.status_code = 200
    client.session.put.return_value.status_code = 200
    return client
