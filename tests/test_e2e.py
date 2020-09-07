import os
import time

import pandas as pd

import taskframe

taskframe.api_key = os.environ.get("TASKFRAME_API_KEY")


class TestClass:
    @classmethod
    def setup_class(cls):
        cls.tf = taskframe.Taskframe.retrieve(os.environ.get("TEST_TASKFRAME_ID"))
        cls.tf.client.session.verify = False

    def test_progress(self):
        data = self.tf.progress()

    def test_add_from_folder(self):
        data = self.tf.progress()
        num_tasks = data["num_tasks"]
        self.tf.add_dataset_from_folder("tests/imgs", pattern="*.jpg")
        self.tf.submit()

        time.sleep(0.2)
        data = self.tf.progress()
        assert data["num_tasks"] == num_tasks + 2

    def test_add_from_list(self):
        data = self.tf.progress()
        num_tasks = data["num_tasks"]
        self.tf.add_dataset_from_list(
            ["tests/imgs/foo.jpg", "tests/imgs/bar.jpg"], custom_ids=[42, 43]
        )

        self.tf.submit()

        time.sleep(0.2)
        data = self.tf.progress()
        assert data["num_tasks"] == num_tasks + 2

    def test_add_from_csv(self):
        data = self.tf.progress()
        num_tasks = data["num_tasks"]
        self.tf.add_dataset_from_csv("tests/img_paths.csv", column="path")

        self.tf.submit()

        time.sleep(0.2)
        data = self.tf.progress()
        assert data["num_tasks"] == num_tasks + 2

    def test_add_from_dataframe(self):
        data = self.tf.progress()
        num_tasks = data["num_tasks"]
        dataframe = pd.read_csv("tests/img_paths.csv")
        self.tf.add_dataset_from_dataframe(dataframe, column="path", base_path="tests")

        self.tf.submit()

        time.sleep(0.2)
        data = self.tf.progress()
        assert data["num_tasks"] == num_tasks + 2

    def test_export(self):
        data = self.tf.progress()
        num_tasks = data["num_tasks"]
        dataframe = pd.read_csv("tests/img_paths.csv")
        self.tf.to_dataframe()

        self.tf.submit()

        time.sleep(0.2)
        data = self.tf.progress()
        assert data["num_tasks"] == num_tasks + 2

    def test_create(self):
        data = self.tf.progress()
        num_tasks = data["num_tasks"]
        dataframe = pd.read_csv("tests/img_paths.csv")
        self.tf.add_dataset_from_dataframe(dataframe, column="path", base_path="tests")

        tf = taskframe.Taskframe(
            data_type="image", task_type="classification", classes=["pos", "neg"]
        )
        tf.submit()

        time.sleep(0.5)
        assert bool(tf.id)
