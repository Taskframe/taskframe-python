import time
from unittest.mock import MagicMock, call, mock_open, patch

import pandas as pd
import pytest

from taskframe import TeamMember
from taskframe.client import API_URL, Client
from taskframe.dataset import CustomIdsLengthMismatch, MissingLabelsMismatch

from .test_utils import custom_mock_open, mock_client, mock_open_func


class TestTeamMemberClass:
    @classmethod
    def setup_class(cls):
        cls.team_member = TeamMember(id="dummy_user_id", taskframe_id="dummy_tf_id")

        cls.team_member_serialized = {
            "id": cls.team_member.id,
            "taskframe_id": cls.team_member.taskframe_id,
            "status": "active",
            "role": "Admin",
            "email": "foo@bar.com",
        }

        TeamMember.client = mock_client()

    def test_retrieve(self):
        team_member = TeamMember.retrieve(
            self.team_member.id, taskframe_id=self.team_member.taskframe_id
        )

        assert isinstance(team_member, TeamMember)

        TeamMember.client.session.get.assert_called_with(
            f"{API_URL}/taskframes/{self.team_member.taskframe_id}/users/{self.team_member.id}/"
        )

    def test_update(self):
        TeamMember.client.session.get.return_value.json.return_value = (
            self.team_member_serialized
        )

        team_member = TeamMember.update(
            self.team_member.id,
            taskframe_id=self.team_member.taskframe_id,
            status="inactive",
        )

        assert isinstance(team_member, TeamMember)

        TeamMember.client.session.put.assert_called_with(
            f"{API_URL}/taskframes/{self.team_member.taskframe_id}/users/{self.team_member.id}/",
            json={
                "id": "dummy_user_id",
                "taskframe_id": "dummy_tf_id",
                "status": "inactive",
                "role": "Admin",
                "email": "foo@bar.com",
            },
        )

    def test_create(self):
        team_member = TeamMember.create(
            taskframe_id=self.team_member.taskframe_id,
            role="Worker",
            email="fizz@buzz.com",
        )

        assert isinstance(team_member, TeamMember)

        TeamMember.client.session.post.assert_called_with(
            f"{API_URL}/taskframes/{self.team_member.taskframe_id}/users/",
            json={
                "id": None,
                "taskframe_id": "dummy_tf_id",
                "email": "fizz@buzz.com",
                "role": "Worker",
                "status": "active",
            },
        )
