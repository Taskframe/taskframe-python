import time
from unittest.mock import MagicMock, call, mock_open, patch

import pandas as pd
import pytest

import taskframe
from taskframe.client import API_URL, Client
from taskframe.dataset import CustomIdsLengthMismatch, MissingLabelsMismatch


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
