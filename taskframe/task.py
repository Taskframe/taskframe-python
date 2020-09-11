from pathlib import Path

from .client import ApiError, Client
from .utils import remove_empty_values


class InvalidParameter(Exception):
    def __init__(self, message="Invalid Parameter"):
        super().__init__(message)


class Task(object):

    client = Client()

    def __init__(
        self,
        id=None,
        custom_id=None,
        taskframe_id=None,
        input_url=None,
        input_data=None,
        input_file=None,
        input_type=None,
        label=None,
        initial_label=None,
        status=None,
    ):

        self.id = id
        self.custom_id = custom_id
        self.taskframe_id = taskframe_id
        self.input_url = input_url
        self.input_data = input_data
        self.input_file = input_file
        self.input_type = input_type
        self.initial_label = initial_label
        self.label = label
        self.status = status

    def __repr__(self):
        return f"<Task object [{self.id}]>"

    @classmethod
    def list(cls, taskframe_id=None, offset=0, limit=25):

        if not taskframe_id:
            raise InvalidParameter(f"Missing required parameter taskframe_id")

        api_resp = cls.client.get(
            f"/tasks/",
            params={"taskframe_id": taskframe_id, "offset": offset, "limit": limit},
        ).json()
        return [cls.from_dict(api_data) for api_data in api_resp["results"]]

    @classmethod
    def retrieve(cls, id=None, custom_id=None, taskframe_id=None):
        api_data = None
        if id:
            api_data = cls.client.get(f"/tasks/{id}/").json()
        elif custom_id and taskframe_id:
            api_resp = cls.client.get(
                f"/tasks/",
                params={"custom_id": custom_id, "taskframe_id": taskframe_id},
            ).json()
            if api_resp["count"] == 1:
                api_data = api_resp["results"][0]
            elif api_resp["count"] == 0:
                raise ApiError(404, {"detail": "Not found."})
            else:
                raise ApiError(400, {"detail": "Multiple objects found."})

        else:
            raise InvalidParameter(f"Missing id or (custom_id,taskframe_id)")
        return cls.from_dict(api_data)

    @classmethod
    def create(
        cls,
        custom_id=None,
        taskframe_id=None,
        input_url=None,
        input_data=None,
        input_file=None,
        initial_label=None,
    ):

        input_params = [input_url, input_data, input_file]

        if sum([bool(x) for x in input_params]) != 1:
            raise InvalidParameter(
                f"One and only one of the following parameters may be specified: input_url, input_data, input_file"
            )

        if not taskframe_id:
            raise InvalidParameter(f"Missing required taskframe_id parameter")

        params = cls(
            custom_id=custom_id,
            taskframe_id=taskframe_id,
            input_url=input_url,
            input_data=input_data,
            input_file=input_file,
            initial_label=initial_label,
        ).to_dict()

        api_data = {}
        if input_file:
            api_data = cls.client.post(f"/tasks/", files=params).json()
        else:
            api_data = cls.client.post(f"/tasks/", json=params).json()
        return cls.from_dict(api_data)

    @classmethod
    def update(cls, id, **kwargs):

        existing_instance = cls.retrieve(id)

        for kwarg, value in kwargs.items():
            if kwarg in [
                "custom_id",
                "initial_label",
                "input_url",
                "input_data",
                "input_file",
            ]:
                setattr(existing_instance, kwarg, value)

        params = existing_instance.to_dict()
        api_data = cls.client.put(f"/tasks/{id}/", json=params).json()
        return cls.from_dict(api_data)

    def submit(self):
        if self.id:
            self.update(
                self.id, custom_id=None, initial_label=self.initial_label,
            )
        else:
            self.create(
                custom_id=self.custom_id,
                taskframe_id=self.taskframe_id,
                input_url=self.input_url,
                input_data=self.input_data,
                input_file=self.input_file,
                initial_label=self.initial_label,
            )

    def dispose(self):
        self.client.post(f"/tasks/{self.id}/dispose/")

    def to_dict(self):

        if self.input_file:
            path = Path(self.input_file)
            file_ = open(path, "rb")
            data = {
                "taskframe_id": (None, self.taskframe_id),
                "input_file": (path.name, file_),
                # "input_type": (None, self.input_type),
            }
            if self.custom_id:
                data["custom_id"] = (None, self.custom_id)
            if self.initial_label:
                data["initial_label"] = (None, self.initial_label)
            return data

        return {
            "id": self.id,
            "custom_id": self.custom_id,
            "taskframe_id": self.taskframe_id,
            "input_url": self.input_url if self.input_url else "",
            "input_data": self.input_data if self.input_data else "",
            "input_file": self.input_file,
            "input_type": self.input_type,
            "initial_label": self.initial_label,
            "label": self.label,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data):

        return cls(
            id=data.get("id"),
            custom_id=data.get("custom_id"),
            taskframe_id=data.get("taskframe_id"),
            input_url=data.get("input_url"),
            input_data=data.get("input_data"),
            input_file=data.get("input_file"),
            input_type=data.get("input_type"),
            label=data.get("label"),
            initial_label=data.get("initial_label"),
            status=data.get("status"),
        )
