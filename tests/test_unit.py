import time
from unittest.mock import MagicMock, call, mock_open, patch

import pandas as pd
import pytest

import taskframe
from taskframe.dataset import CustomIdsMismatch


def mock_open_func(filename, *args, **kwargs):
    return str(filename)


custom_mock_open = mock_open()
custom_mock_open.side_effect = mock_open_func


class TestClass:
    @classmethod
    def setup_class(cls):
        cls.tf = taskframe.Taskframe(id="dummy_id")
        cls.tf.session = MagicMock()
        cls.tf.session.post.return_value.status_code = 200
        with patch("taskframe.src.open_file", custom_mock_open) as m_open:
            cls.calls = [
                call(
                    f"{taskframe.API_URL}/tasks/",
                    files={
                        "taskframe_id": (None, cls.tf.id),
                        "input_file": ("foo.jpg", m_open("tests/imgs/foo.jpg", "rb"),),
                        "custom_id": (None, 42),
                    },
                ),
                call(
                    f"{taskframe.API_URL}/tasks/",
                    files={
                        "taskframe_id": (None, cls.tf.id),
                        "input_file": ("bar.jpg", m_open("tests/imgs/bar.jpg", "rb"),),
                        "custom_id": (None, 43),
                    },
                ),
            ]

            cls.calls_str_custom_id = [
                call(
                    f"{taskframe.API_URL}/tasks/",
                    files={
                        "taskframe_id": (None, cls.tf.id),
                        "input_file": ("foo.jpg", m_open("tests/imgs/foo.jpg", "rb"),),
                        "custom_id": (None, "foo"),
                    },
                ),
                call(
                    f"{taskframe.API_URL}/tasks/",
                    files={
                        "taskframe_id": (None, cls.tf.id),
                        "input_file": ("bar.jpg", m_open("tests/imgs/bar.jpg", "rb"),),
                        "custom_id": (None, "bar"),
                    },
                ),
            ]

        cls.urls = [
            "https://i.dailymail.co.uk/1s/2019/11/23/09/21370544-0-image-a-4_1574501241272.jpg",
            "https://images.pexels.com/photos/104827/cat-pet-animal-domestic-104827.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
        ]

        cls.urls_json_data = [
            {
                "taskframe_id": cls.tf.id,
                "custom_id": "fizz",
                "input_url": "https://i.dailymail.co.uk/1s/2019/11/23/09/21370544-0-image-a-4_1574501241272.jpg",
            },
            {
                "taskframe_id": cls.tf.id,
                "custom_id": "buzz",
                "input_url": "https://images.pexels.com/photos/104827/cat-pet-animal-domestic-104827.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
            },
        ]

    def test_sync(self):
        data = self.tf.sync()

        self.tf.session.get.assert_called_with(
            f"{taskframe.API_URL}/taskframes/{self.tf.id}/"
        )

    @patch("taskframe.src.open_file", custom_mock_open)
    def test_add_from_folder(self):
        self.tf.add_dataset_from_folder("tests/imgs", custom_ids=[42, 43])
        self.tf.submit()
        self.tf.session.post.assert_has_calls(self.calls, any_order=True)

    @patch("taskframe.src.open_file", custom_mock_open)
    def test_add_from_list(self):

        with pytest.raises(CustomIdsMismatch) as exception:
            self.tf.add_dataset_from_list(
                ["tests/imgs/foo.jpg", "tests/imgs/bar.jpg"], custom_ids=[42, 43, 44]
            )

        self.tf.add_dataset_from_list(
            ["tests/imgs/foo.jpg", "tests/imgs/bar.jpg"], custom_ids=[42, 43]
        )

        self.tf.submit()
        self.tf.session.post.assert_has_calls(self.calls, any_order=True)

    @patch("taskframe.src.open_file", custom_mock_open)
    def test_add_from_csv(self):
        self.tf.add_dataset_from_csv(
            "tests/img_paths.csv", column="path", custom_id_column="identifier"
        )
        self.tf.submit()
        self.tf.session.post.assert_has_calls(self.calls_str_custom_id, any_order=True)

    @patch("taskframe.src.open_file", custom_mock_open)
    def test_add_from_dataframe(self):
        dataframe = pd.read_csv("tests/img_paths.csv")
        self.tf.add_dataset_from_dataframe(
            dataframe, column="path", base_path="tests", custom_id_column="identifier"
        )
        self.tf.submit()
        self.tf.session.post.assert_has_calls(self.calls_str_custom_id, any_order=True)

    @patch("taskframe.src.open_file", custom_mock_open)
    def test_add_urls_from_list(self):

        with pytest.raises(CustomIdsMismatch) as exception:
            self.tf.add_dataset_from_list(self.urls, custom_ids=[42, 43, 44])

        self.tf.add_dataset_from_list(
            self.urls, custom_ids=["fizz", "buzz"],
        )

        self.tf.submit()
        self.tf.session.post.assert_called_with(
            f"{taskframe.API_URL}/tasks/", json=self.urls_json_data
        )

    @patch("taskframe.src.open_file", custom_mock_open)
    def test_add_urls_from_csv(self):

        self.tf.add_dataset_from_csv(
            "tests/img_urls.csv", column="url", custom_id_column="identifier"
        )

        self.tf.submit()
        self.tf.session.post.assert_called_with(
            f"{taskframe.API_URL}/tasks/", json=self.urls_json_data
        )

    @patch("taskframe.src.open_file", custom_mock_open)
    def test_add_urls_from_dataframe(self):
        dataframe = pd.read_csv("tests/img_urls.csv")
        self.tf.add_dataset_from_dataframe(
            dataframe, column="url", custom_id_column="identifier"
        )

        self.tf.submit()
        self.tf.session.post.assert_called_with(
            f"{taskframe.API_URL}/tasks/", json=self.urls_json_data
        )
