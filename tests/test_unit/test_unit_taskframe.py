import time
from unittest.mock import MagicMock, call, mock_open, patch

import pandas as pd
import pytest

from taskframe.client import API_URL, Client
from taskframe.taskframe import InvalidParameter, Taskframe
from taskframe.team_member import TeamMember

from .test_utils import custom_mock_open, mock_client, mock_open_func


class TestClass:
    @classmethod
    def setup_class(cls):
        cls.tf = Taskframe(id="dummy_id")

        cls.tf_serialized = {
            "id": "dummy_id",
            "name": "",
            "data_type": "image",
            "task_type": "classification",
            "mode": "inhouse",
            "output_schema": None,
            "output_schema_url": "",
            "ui_schema": None,
            "ui_schema_url": "",
            "params": {},
            "instructions": "",
            "redundancy": 1,
            "review": True,
            "callback_url": "",
        }
        Taskframe.client = mock_client()

        TeamMember.client = mock_client()

    def test_retrieve(self):
        tf = Taskframe.retrieve(self.tf.id)

        Taskframe.client.session.get.assert_called_with(
            f"{API_URL}/taskframes/{self.tf.id}/"
        )

    def test_list(self):
        Taskframe.client.session.get.return_value.json.return_value = {
            "count": 2,
            "next": "...",
            "previous": None,
            "results": [
                {
                    "id": "abcd123",
                    "name": "",
                    "data_type": "image",
                    "task_type": "classification",
                    "num_tasks": 0,
                    "num_assignable": 0,
                    "num_pending_work": 0,
                    "num_pending_review": 0,
                    "num_finished": 0,
                    "mode": "inhouse",
                    "env": "sandbox",
                },
                {
                    "id": "efgh456",
                    "name": "",
                    "data_type": "text",
                    "task_type": "classification",
                    "num_tasks": 0,
                    "num_assignable": 0,
                    "num_pending_work": 0,
                    "num_pending_review": 0,
                    "num_finished": 0,
                    "mode": "inhouse",
                    "env": "sandbox",
                },
            ],
        }
        tfs = Taskframe.list()

        Taskframe.client.session.get.assert_called_with(
            f"{API_URL}/taskframes/", params={"offset": 0, "limit": 25}
        )

        assert len(tfs) == 2

        assert all([isinstance(tf, Taskframe) for tf in tfs])

    def test_update(self):
        Taskframe.client.session.get.return_value.json.return_value = self.tf_serialized

        tf = Taskframe.update(
            self.tf.id, classes=["fizz", "buzz"], name="this is the name",
        )

        assert isinstance(tf, Taskframe)

        updated_tf_serialized = self.tf_serialized.copy()

        updated_tf_serialized["name"] = "this is the name"
        updated_tf_serialized["params"]["classes"] = ["fizz", "buzz"]

        Taskframe.client.session.put.assert_called_with(
            f"{API_URL}/taskframes/{self.tf.id}/", json=updated_tf_serialized,
        )

    def test_create(self):
        with pytest.raises(InvalidParameter) as exception:
            tf = Taskframe.create(
                data_type="text",
                task_type="classification",
                classes=["foo", "bar"],
                invalid_param="invalid",
            )

        tf = Taskframe.create(
            data_type="text", task_type="classification", classes=["foo", "bar"],
        )

        Taskframe.client.session.post.assert_called_with(
            f"{API_URL}/taskframes/",
            json={
                "id": None,
                "name": "",
                "data_type": "text",
                "task_type": "classification",
                "params": {"classes": ["foo", "bar"]},
                "output_schema": None,
                "output_schema_url": "",
                "ui_schema": None,
                "ui_schema_url": "",
                "instructions": "",
                "mode": "inhouse",
                "redundancy": 1,
                "review": True,
                "callback_url": "",
            },
        )

    def test_submit(self):
        tf = Taskframe(
            data_type="text", task_type="classification", classes=["foo", "bar"],
        )

        tf.submit()

        Taskframe.client.session.post.assert_called_with(
            f"{API_URL}/taskframes/",
            json={
                "id": None,
                "name": "",
                "data_type": "text",
                "task_type": "classification",
                "params": {"classes": ["foo", "bar"]},
                "output_schema": None,
                "output_schema_url": "",
                "ui_schema": None,
                "ui_schema_url": "",
                "instructions": "",
                "mode": "inhouse",
                "redundancy": 1,
                "review": True,
                "callback_url": "",
            },
        )

    def test_add_team(self):
        self.tf.add_team(
            workers=["prexisting@worker.com"], reviewers=["sam@reviewer.com"],
        )

        TeamMember.client.session.get.return_value.json.side_effect = [
            # first call in TeamMember.list
            {
                "results": [
                    {
                        "email": "prexisting@worker.com",
                        "role": "worker",
                        "id": 1232442,
                        "status": "inactive",
                    }
                ]
            },
            # second call in TeamMember.retrieve before the update
            {
                "email": "prexisting@worker.com",
                "role": "worker",
                "id": 1232442,
                "status": "inactive",  # should be reactivated
            },
        ]

        self.tf.submit_team()

        TeamMember.client.session.post.assert_called_with(
            f"{API_URL}/taskframes/{self.tf.id}/users/",
            json={
                "role": "reviewer",
                "status": "active",
                "email": "sam@reviewer.com",
                "taskframe_id": self.tf.id,
                "id": None,
            },
        )

        TeamMember.client.session.put.assert_called_with(
            f"{API_URL}/taskframes/{self.tf.id}/users/1232442/",
            json={
                "id": 1232442,
                "taskframe_id": self.tf.id,
                "email": "prexisting@worker.com",
                "role": "worker",
                "status": "active",
            },
        )
