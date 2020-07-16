import time
from unittest.mock import MagicMock, call, mock_open, patch

import pandas as pd
import pytest

import taskframe
from taskframe.client import API_URL
from taskframe.dataset import CustomIdsLengthMismatch


def mock_open_func(filename, *args, **kwargs):
    return str(filename)


custom_mock_open = mock_open()
custom_mock_open.side_effect = mock_open_func


class TestClass:
    @classmethod
    def setup_class(cls):
        cls.tf = taskframe.Taskframe(id="dummy_id")
        cls.tf.client.session = MagicMock()
        cls.tf.client.session.post.return_value.status_code = 200
        cls.tf.client.session.get.return_value.status_code = 200
        cls.tf.client.session.put.return_value.status_code = 200
        with patch("taskframe.dataset.open_file", custom_mock_open) as m_open:
            cls.calls = [
                call(
                    f"{API_URL}/tasks/",
                    files={
                        "taskframe_id": (None, cls.tf.id),
                        "input_file": ("foo.jpg", m_open("tests/imgs/foo.jpg", "rb"),),
                        "custom_id": (None, 42),
                        "input_type": (None, "file"),
                        "is_training": (None, False),
                    },
                ),
                call(
                    f"{API_URL}/tasks/",
                    files={
                        "taskframe_id": (None, cls.tf.id),
                        "input_file": ("bar.jpg", m_open("tests/imgs/bar.jpg", "rb"),),
                        "input_type": (None, "file"),
                        "custom_id": (None, 43),
                        "is_training": (None, True),
                        "label": (None, '"cat"'),
                    },
                ),
            ]

            cls.calls_str_custom_id = [
                call(
                    f"{API_URL}/tasks/",
                    files={
                        "taskframe_id": (None, cls.tf.id),
                        "input_file": ("foo.jpg", m_open("tests/imgs/foo.jpg", "rb"),),
                        "input_type": (None, "file"),
                        "custom_id": (None, "foo"),
                        "is_training": (None, False),
                    },
                ),
                call(
                    f"{API_URL}/tasks/",
                    files={
                        "taskframe_id": (None, cls.tf.id),
                        "input_file": ("bar.jpg", m_open("tests/imgs/bar.jpg", "rb"),),
                        "custom_id": (None, "bar"),
                        "input_type": (None, "file"),
                        "is_training": (None, True),
                        "label": (None, '"cat"'),
                    },
                ),
            ]

        cls.calls_str_custom_id = [
            call(
                f"{API_URL}/tasks/",
                files={
                    "taskframe_id": (None, cls.tf.id),
                    "input_file": ("foo.jpg", m_open("tests/imgs/foo.jpg", "rb"),),
                    "input_type": (None, "file"),
                    "custom_id": (None, "foo"),
                    "is_training": (None, False),
                },
            ),
            call(
                f"{API_URL}/tasks/",
                files={
                    "taskframe_id": (None, cls.tf.id),
                    "input_file": ("bar.jpg", m_open("tests/imgs/bar.jpg", "rb"),),
                    "custom_id": (None, "bar"),
                    "input_type": (None, "file"),
                    "is_training": (None, True),
                    "label": (None, '"cat"'),
                },
            ),
        ]

        cls.urls = [
            "https://i.dailymail.co.uk/1s/2019/11/23/09/21370544-0-image-a-4_1574501241272.jpg",
            "https://images.pexels.com/photos/104827/cat-pet-animal-domestic-104827.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
        ]

        cls.urls_json_data = {
            "items": [
                {
                    "taskframe_id": cls.tf.id,
                    "custom_id": "fizz",
                    "input_url": "https://i.dailymail.co.uk/1s/2019/11/23/09/21370544-0-image-a-4_1574501241272.jpg",
                    "input_type": "url",
                    "is_training": False,
                },
                {
                    "taskframe_id": cls.tf.id,
                    "custom_id": "buzz",
                    "input_url": "https://images.pexels.com/photos/104827/cat-pet-animal-domestic-104827.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
                    "is_training": True,
                    "input_type": "url",
                    "label": "cat",
                },
            ]
        }

    def test_fetch(self):
        data = self.tf.fetch()

        self.tf.client.session.get.assert_called_with(
            f"{API_URL}/taskframes/{self.tf.id}/"
        )

    @patch("taskframe.dataset.open_file", custom_mock_open)
    def test_add_from_list(self):

        with pytest.raises(CustomIdsLengthMismatch) as exception:
            self.tf.add_dataset_from_list(
                ["tests/imgs/foo.jpg", "tests/imgs/bar.jpg"], custom_ids=[42, 43, 44],
            )

        self.tf.add_dataset_from_list(
            ["tests/imgs/foo.jpg", "tests/imgs/bar.jpg"],
            custom_ids=[42, 43],
            labels=[None, "cat"],
        )

        self.tf.submit()

        self.tf.client.session.post.assert_called_with(
            f"{API_URL}/tasks/",
            files={
                "taskframe_id": (None, self.tf.id),
                "input_file": ("bar.jpg", mock_open_func("tests/imgs/bar.jpg", "rb"),),
                "input_type": (None, "file"),
                "custom_id": (None, 43),
                "is_training": (None, True),
                "label": (None, '"cat"'),
            },
        )

        self.tf.client.session.post.assert_has_calls(self.calls, any_order=True)

    @patch("taskframe.dataset.open_file", custom_mock_open)
    def test_add_from_folder(self):
        self.tf.add_dataset_from_folder(
            "tests/imgs", custom_ids=[42, 43], labels=[None, "cat"], pattern="*.jpg"
        )

        assert len(self.tf.dataset) == 2

        self.tf.submit()
        self.tf.client.session.post.assert_has_calls(self.calls, any_order=True)

    @patch("taskframe.dataset.open_file", custom_mock_open)
    def test_add_from_folder_recursive(self):
        self.tf.add_dataset_from_folder("tests/imgs", recursive=True, pattern="*.jpg")

        assert len(self.tf.dataset) == 3

    @patch("taskframe.dataset.open_file", custom_mock_open)
    def test_add_from_csv(self):
        self.tf.add_dataset_from_csv(
            "tests/img_paths.csv",
            column="path",
            custom_id_column="identifier",
            label_column="label",
        )
        self.tf.submit()
        self.tf.client.session.post.assert_has_calls(
            self.calls_str_custom_id, any_order=True
        )

    @patch("taskframe.dataset.open_file", custom_mock_open)
    def test_add_from_dataframe(self):
        dataframe = pd.read_csv("tests/img_paths.csv")
        self.tf.add_dataset_from_dataframe(
            dataframe,
            column="path",
            base_path="tests",
            custom_id_column="identifier",
            label_column="label",
        )
        self.tf.submit()

        self.tf.client.session.post.assert_called_with(
            f"{API_URL}/tasks/",
            files={
                "taskframe_id": (None, self.tf.id),
                "input_file": ("bar.jpg", mock_open_func("tests/imgs/bar.jpg", "rb"),),
                "custom_id": (None, "bar"),
                "input_type": (None, "file"),
                "is_training": (None, True),
                "label": (None, '"cat"'),
            },
        )

        self.tf.client.session.post.assert_has_calls(
            self.calls_str_custom_id, any_order=True
        )

    @patch("taskframe.dataset.open_file", custom_mock_open)
    def test_add_urls_from_list(self):

        with pytest.raises(CustomIdsLengthMismatch) as exception:
            self.tf.add_dataset_from_list(
                self.urls, custom_ids=[42, 43, 44],
            )

        self.tf.add_dataset_from_list(
            self.urls, custom_ids=["fizz", "buzz"], labels=[None, "cat"]
        )

        self.tf.submit()
        self.tf.client.session.post.assert_called_with(
            f"{API_URL}/tasks/",
            json=self.urls_json_data,
            params={"taskframe_id": self.tf.id},
        )

    @patch("taskframe.dataset.open_file", custom_mock_open)
    def test_add_urls_from_csv(self):

        self.tf.add_dataset_from_csv(
            "tests/img_urls.csv",
            column="url",
            custom_id_column="identifier",
            label_column="label",
        )

        self.tf.submit()
        self.tf.client.session.post.assert_called_with(
            f"{API_URL}/tasks/",
            json=self.urls_json_data,
            params={"taskframe_id": self.tf.id},
        )

    @patch("taskframe.dataset.open_file", custom_mock_open)
    def test_add_urls_from_dataframe(self):
        dataframe = pd.read_csv("tests/img_urls.csv")
        self.tf.add_dataset_from_dataframe(
            dataframe, column="url", custom_id_column="identifier", label_column="label"
        )

        self.tf.submit()
        self.tf.client.session.post.assert_called_with(
            f"{API_URL}/tasks/",
            json=self.urls_json_data,
            params={"taskframe_id": self.tf.id},
        )
