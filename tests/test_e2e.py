import os
import time
import uuid

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
        self.tf.dataset = None
        self.tf.trainingset = None
        data = self.tf.progress()
        num_tasks = data["num_tasks"]
        self.tf.add_dataset_from_folder("tests/imgs", pattern="*.jpg")
        self.tf.submit()

        time.sleep(0.2)
        data = self.tf.progress()
        assert data["num_tasks"] == num_tasks + 2

    def test_add_training_from_list(self):
        self.tf.dataset = None
        self.tf.trainingset = None
        data = self.tf.progress()

        num_tasks = data["num_tasks"]
        self.tf.add_trainingset_from_list(
            ["tests/imgs/foo.jpg", "tests/imgs/bar.jpg"],
            custom_ids=[rand(), rand()],
            labels=["dog", "cat"],
        )

        self.tf.submit()

        time.sleep(0.2)
        data = self.tf.progress()
        assert data["num_tasks"] == num_tasks

    def test_add_from_list(self):
        self.tf.dataset = None
        self.tf.trainingset = None
        data = self.tf.progress()
        num_tasks = data["num_tasks"]

        custom_ids = [rand(), rand()]

        self.tf.add_dataset_from_list(
            ["tests/imgs/foo.jpg", "tests/imgs/bar.jpg"], custom_ids=custom_ids
        )

        self.tf.submit()

        time.sleep(0.2)
        data = self.tf.progress()
        assert data["num_tasks"] == num_tasks + 2

    def test_add_from_csv(self):
        self.tf.dataset = None
        self.tf.trainingset = None
        data = self.tf.progress()
        num_tasks = data["num_tasks"]
        self.tf.add_dataset_from_csv("tests/img_paths.csv", column="path")

        self.tf.submit()

        time.sleep(0.2)
        data = self.tf.progress()
        assert data["num_tasks"] == num_tasks + 2

    def test_add_from_dataframe(self):
        self.tf.dataset = None
        self.tf.trainingset = None
        data = self.tf.progress()
        num_tasks = data["num_tasks"]
        dataframe = pd.read_csv("tests/img_paths.csv")
        self.tf.add_dataset_from_dataframe(dataframe, column="path", base_path="tests")

        self.tf.submit()

        time.sleep(0.2)
        data = self.tf.progress()
        assert data["num_tasks"] == num_tasks + 2

    def test_export(self):
        self.tf.dataset = None
        self.tf.trainingset = None

        self.tf.to_dataframe()

    def test_create(self):
        self.tf.dataset = None
        self.tf.trainingset = None
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

    def test_task_class(self):
        custom_id = rand()

        taskframe.Task.create(
            custom_id=custom_id,
            taskframe_id=self.tf.id,
            input_file="tests/imgs/foo.jpg",
        )

        task = taskframe.Task.retrieve(custom_id=custom_id, taskframe_id=self.tf.id)

        assert task.input_type == "file"
        assert task.input_url == ""

        new_custom_id = rand()
        taskframe.Task.update(
            task.id, custom_id=new_custom_id, input_url="http://example.com/img.jpg",
        )

        task = taskframe.Task.retrieve(custom_id=new_custom_id, taskframe_id=self.tf.id)

        assert task.input_url == "http://example.com/img.jpg"
        assert task.input_type == "url"

        taskframe.Task.update(
            task.id, input_file="tests/imgs/bar.jpg",
        )

        task = taskframe.Task.retrieve(custom_id=new_custom_id, taskframe_id=self.tf.id)
        assert task.input_url == ""
        assert task.input_type == "file"

    def test_team_member_class(self):
        member_id = taskframe.TeamMember.create(
            taskframe_id=self.tf.id, email=f"{rand()}@{rand()}.com", role="worker",
        ).id
        member = taskframe.TeamMember.retrieve(id=member_id, taskframe_id=self.tf.id)

        assert member.status == "active"

        taskframe.TeamMember.update(
            member_id, taskframe_id=self.tf.id, role="reviewer", status="inactive"
        )

        member = taskframe.TeamMember.retrieve(id=member_id, taskframe_id=self.tf.id)

        assert member.status == "inactive"
        assert member.role == "reviewer"

    def test_export(self):
        df = self.tf.to_dataframe()
        self.tf.to_csv("dev/test_e2e_export.csv")

    def test_merge_to_dataframe(self):
        initial_dataframe = pd.read_csv("tests/img_paths.csv")[["path", "identifier"]]
        labelled_dataframe = self.tf.merge_to_dataframe(
            initial_dataframe, custom_id_column="identifier"
        )

        # label column already present : should be dropped.
        initial_dataframe = pd.read_csv("tests/img_paths.csv")[
            ["path", "identifier", "label"]
        ]
        labelled_dataframe = self.tf.merge_to_dataframe(
            initial_dataframe, custom_id_column="identifier"
        )


def rand(n=6):
    return uuid.uuid4().hex[:n]
