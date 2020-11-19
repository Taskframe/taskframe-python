from pathlib import Path

from .client import ApiError, Client


class InvalidParameter(Exception):
    def __init__(self, message="Invalid Parameter"):
        super().__init__(message)


class Task(object):

    client = Client()

    input_fields = ["input_url", "input_data", "input_file"]

    def __init__(
        self,
        id=None,
        custom_id=None,
        taskframe_id=None,
        input_url="",
        input_data="",
        input_file=None,
        input_type=None,
        label=None,
        initial_label=None,
        status="pending_work",
        priority=None,
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
        self.priority = priority

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
        input_url="",
        input_data="",
        input_file=None,
        initial_label=None,
        priority=None,
    ):

        input_params = [input_url, input_data, input_file]

        if sum([bool(x) for x in input_params]) != 1:
            raise InvalidParameter(
                f"One and only one of the following parameters may be specified: input_url, input_data, input_file"
            )

        if not taskframe_id:
            raise InvalidParameter(f"Missing required taskframe_id parameter")

        api_params = cls(
            custom_id=custom_id,
            taskframe_id=taskframe_id,
            input_url=input_url,
            input_data=input_data,
            input_file=input_file,
            initial_label=initial_label,
            priority=priority,
        ).to_api_params()

        api_data = cls.client.post("/tasks/", **api_params).json()
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
                "priority",
            ]:
                setattr(existing_instance, kwarg, value)

            if kwarg in cls.input_fields and value:
                # unset other input_fields
                existing_instance.input_type = None
                other_input_fields = [x for x in cls.input_fields if x != kwarg]
                for other_input_field in other_input_fields:
                    empty_val = None if other_input_field == "input_file" else ""
                    setattr(existing_instance, other_input_field, empty_val)

        api_params = existing_instance.to_api_params()
        api_data = cls.client.put(f"/tasks/{id}/", **api_params).json()
        return cls.from_dict(api_data)

    def submit(self):
        if self.id:
            self.update(
                self.id,
                custom_id=None,
                initial_label=self.initial_label,
            )
        else:
            self.create(
                custom_id=self.custom_id,
                taskframe_id=self.taskframe_id,
                input_url=self.input_url,
                input_data=self.input_data,
                input_file=self.input_file,
                initial_label=self.initial_label,
                priority=self.priority,
            )

    def dispose(self):
        self.client.post(f"/tasks/{self.id}/dispose/")

    def to_dict(self):
        return {
            "id": self.id,
            "custom_id": self.custom_id,
            "taskframe_id": self.taskframe_id,
            "input_url": self.input_url,
            "input_data": self.input_data,
            "input_file": self.input_file,
            "input_type": self.input_type,
            "initial_label": self.initial_label,
            "label": self.label,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data):

        return cls(
            id=data.get("id"),
            custom_id=data.get("custom_id"),
            taskframe_id=data.get("taskframe_id"),
            input_url=data.get("input_url", ""),
            input_data=data.get("input_data", ""),
            input_file=data.get("input_file"),
            input_type=data.get("input_type"),
            label=data.get("label"),
            initial_label=data.get("initial_label"),
            status=data.get("status"),
            priority=data.get("priority"),
        )

    def to_api_params(self):
        dict_data = self.to_dict()
        if not self.input_file:
            return {"json": dict_data}

        path = Path(self.input_file)
        file_ = open(path, "rb")
        dict_data.pop("input_file")

        return {
            "files": {
                "input_file": (path.name, file_),
            },
            "data": dict_data,
        }
