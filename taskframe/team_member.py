from pathlib import Path

from .client import ApiError, Client
from .utils import remove_empty_values


class InvalidParameter(Exception):
    def __init__(self, message="Invalid Parameter"):
        super().__init__(message)


class TeamMember(object):

    ROLE_CHOICES = ["Admin", "Worker", "Reviewer"]
    STATUS_CHOICES = ["active", "inactive"]

    client = Client()

    def __init__(
        self, id=None, taskframe_id=None, role=None, status="active", email=None,
    ):

        self.id = id
        self.taskframe_id = taskframe_id
        self.role = role
        self.status = status
        self.email = email

    def __repr__(self):
        return f"<TeamMember object [{self.email} {self.role} {self.status}]>"

    @classmethod
    def list(cls, taskframe_id=None, offset=0, limit=25):

        if not taskframe_id:
            raise InvalidParameter(f"Missing required parameter taskframe_id")

        api_resp = cls.client.get(
            f"/taskframes/{taskframe_id}/users/",
            params={"offset": offset, "limit": limit},
        ).json()
        return [cls.from_dict(api_data) for api_data in api_resp["results"]]

    @classmethod
    def retrieve(cls, id=None, taskframe_id=None):
        if not any([bool(x) for x in [id, taskframe_id]]):
            raise InvalidParameter(
                f"Missing required parameter. Required parameters: id, taskframe_id"
            )

        api_data = cls.client.get(f"/taskframes/{taskframe_id}/users/{id}/").json()
        api_data["taskframe_id"] = taskframe_id
        return cls.from_dict(api_data)

    @classmethod
    def create(cls, taskframe_id=None, email=None, role=None, status="active"):

        if not any([bool(x) for x in [taskframe_id, email, role]]):
            raise InvalidParameter(
                f"Missing required parameter. Required parameters: taskframe_id, email, role"
            )

        cls.check_params(role, status)

        params = cls(
            taskframe_id=taskframe_id, email=email, role=role, status=status
        ).to_dict()

        api_data = cls.client.post(
            f"/taskframes/{taskframe_id}/users/", json=params
        ).json()
        return cls.from_dict(api_data)

    @classmethod
    def update(cls, id, taskframe_id=None, role=None, status=None):

        if not all([bool(x) for x in [id, taskframe_id]]):
            raise InvalidParameter(
                f"Missing required parameter. Required parameters: id, taskframe_id"
            )

        cls.check_params(role, status)
        existing_instance = cls.retrieve(id, taskframe_id=taskframe_id)
        if role:
            existing_instance.role = role
        if status:
            existing_instance.status = status

        params = existing_instance.to_dict()

        api_data = cls.client.put(
            f"/taskframes/{taskframe_id}/users/{id}/", json=params
        ).json()
        return cls.from_dict(api_data)

    def submit(self):
        if self.id:
            self.update(
                self.id,
                taskframe_id=self.taskframe_id,
                role=self.role,
                status=self.status,
            )
        else:
            self.create(
                taskframe_id=self.taskframe_id,
                email=self.email,
                role=self.role,
                status=self.status,
            )

    @classmethod
    def check_params(cls, role, status):
        if role and role not in cls.ROLE_CHOICES:
            raise InvalidParameter(
                f"Invalid role {role}, valid roles are: {', '.join(cls.ROLE_CHOICES)}"
            )
        if status and status not in cls.STATUS_CHOICES:
            raise InvalidParameter(
                f"Invalid status {status}, valid statuses are:  {', '.join(cls.STATUS_CHOICES)}"
            )

    def to_dict(self):
        return {
            "id": self.id,
            "taskframe_id": self.taskframe_id,
            "email": self.email,
            "role": self.role,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data):

        return cls(
            id=data.get("id"),
            taskframe_id=data.get("taskframe_id"),
            email=data.get("email"),
            role=data.get("role"),
            status=data.get("status", "active"),
        )
