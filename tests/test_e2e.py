import os
import time

import pandas as pd

import taskframe

taskframe.api_key = os.environ.get("TASKFRAME_API_KEY")


class TestClass:
    @classmethod
    def setup_class(cls):
        cls.tf = taskframe.Taskframe(id=os.environ.get("TEST_TASKFRAME_ID"))
        cls.tf.session.verify = False

    def test_sync(self):
        data = self.tf.sync()

    def test_add_from_folder(self):
        data = self.tf.sync()
        num_tasks = data["num_tasks"]
        self.tf.add_dataset_from_folder("tests/imgs")
        self.tf.submit()

        time.sleep(0.5)
        data = self.tf.sync()
        assert data["num_tasks"] == num_tasks + 2

    def test_add_from_list(self):
        data = self.tf.sync()
        num_tasks = data["num_tasks"]
        self.tf.add_dataset_from_list(
            ["tests/imgs/foo.jpg", "tests/imgs/bar.jpg"], custom_ids=[42, 43]
        )

        self.tf.submit()

        time.sleep(0.5)
        data = self.tf.sync()
        assert data["num_tasks"] == num_tasks + 2

    def test_add_from_csv(self):
        data = self.tf.sync()
        num_tasks = data["num_tasks"]
        self.tf.add_dataset_from_csv("tests/img_paths.csv", column="path")

        self.tf.submit()

        time.sleep(0.5)
        data = self.tf.sync()
        assert data["num_tasks"] == num_tasks + 2

    def test_add_from_dataframe(self):
        data = self.tf.sync()
        num_tasks = data["num_tasks"]
        dataframe = pd.read_csv("tests/img_paths.csv")
        self.tf.add_dataset_from_dataframe(dataframe, column="path", base_path="tests")

        self.tf.submit()

        time.sleep(0.5)
        data = self.tf.sync()
        assert data["num_tasks"] == num_tasks + 2
