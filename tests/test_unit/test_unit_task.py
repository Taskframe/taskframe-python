from taskframe import Task
from taskframe.client import API_URL

from .test_utils import mock_client


class TestTaskClass:
    @classmethod
    def setup_class(cls):
        cls.task = Task(id="dummy_task_id", taskframe_id="dummy_tf_id")

        cls.task_serialized = {
            "id": cls.task.id,
            "taskframe_id": cls.task.taskframe_id,
            "custom_id": "mycustomid",
            "initial_label": "myinitiallabel",
            "input_url": "http://example.com/img.jpg",
            "input_type": "url",
        }

        Task.client = mock_client()

    def test_retrieve(self):
        task = Task.retrieve(self.task.id)

        assert isinstance(task, Task)

        Task.client.session.get.assert_called_with(f"{API_URL}/tasks/{self.task.id}/")

    def test_update(self):
        Task.client.session.get.return_value.json.return_value = {
            "id": self.task.id,
            "taskframe_id": self.task.taskframe_id,
            "custom_id": "mycustomid",
            "initial_label": "myinitiallabel",
            "input_url": "http://example.com/img.jpg",
            "input_type": "url",
            "status": "pending_work",
            "label": None,
            "priority": None,
        }

        task = Task.update(
            self.task.id,
            custom_id="mycustomid2",
        )

        assert isinstance(task, Task)

        Task.client.session.put.assert_called_with(
            f"{API_URL}/tasks/{self.task.id}/",
            json={
                "id": "dummy_task_id",
                "taskframe_id": "dummy_tf_id",
                "custom_id": "mycustomid2",
                "initial_label": "myinitiallabel",
                "input_url": "http://example.com/img.jpg",
                "input_type": "url",
                "input_file": None,
                "input_data": "",
                "label": None,
                "priority": None,
            },
        )

    def test_create(self):
        task = Task.create(
            taskframe_id=self.task.taskframe_id,
            input_url="http://foo.com/img.jpg",
            initial_label="foo",
        )

        assert isinstance(task, Task)

        Task.client.session.post.assert_called_with(
            f"{API_URL}/tasks/",
            json={
                "id": None,
                "custom_id": None,
                "taskframe_id": "dummy_tf_id",
                "input_url": "http://foo.com/img.jpg",
                "input_data": "",
                "input_file": None,
                "input_type": None,
                "initial_label": "foo",
                "label": None,
                "priority": None,
            },
        )
