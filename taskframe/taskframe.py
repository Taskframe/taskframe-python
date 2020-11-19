import csv
import json
import os
import random
from warnings import warn

from IPython.display import HTML, Javascript, display

from .client import Client
from .dataset import Dataset, Trainingset
from .team_member import TeamMember

APP_ENDPOINT = os.environ.get("TASKFRAME_APP_ENDPOINT", "https://app.taskframe.ai")


class CustomIdsMismatch(Exception):
    def __init__(self, message="mismatch in length of dataset and custom_ids"):
        super().__init__(message)


class InvalidParameter(Exception):
    def __init__(self, message="Invalid Parameter"):
        super().__init__(message)


class MissingId(Exception):
    def __init__(
        self, message="Missing taskframe id. You probably need to call submit() first."
    ):
        super().__init__(message)


class Taskframe(object):

    client = Client()

    acceptable_params = [
        "classes",
        "tags",
        "global_classes",
        "global_tags",
        "global_json_schema",
        "global_ui_schema",
        "multiple",
        "files_accepted",
    ]

    def __init__(
        self,
        data_type=None,
        task_type=None,
        json_schema=None,
        json_schema_url="",
        ui_schema=None,
        ui_schema_url="",
        instructions="",
        name="",
        id=None,
        review=True,
        redundancy=1,
        callback_url="",
        **kwargs,
    ):
        self.data_type = data_type
        self.task_type = task_type
        self.json_schema = json_schema
        self.json_schema_url = json_schema_url
        self.ui_schema = ui_schema
        self.ui_schema_url = ui_schema_url
        self.instructions = instructions
        self.name = name
        self.id = id
        self.dataset = None
        self.trainingset = None
        self.team = None
        self.review = review
        self.redundancy = redundancy
        self.callback_url = callback_url
        self.workers = []
        self.reviewers = []

        self._check_params(kwargs)

        self.kwargs = kwargs

    def __repr__(self):
        return f"<Taskframe object {self.id}[{self.data_type} {self.task_type}]>"

    @classmethod
    def list(cls, offset=0, limit=25):
        api_resp = cls.client.get(
            f"/taskframes/", params={"offset": offset, "limit": limit}
        ).json()
        return [cls.from_dict(api_data) for api_data in api_resp.get("results", [])]

    @classmethod
    def retrieve(cls, id):
        """Sync method to get a Taskframe from the API"""
        api_data = cls.retrieve_data(id)
        return cls.from_dict(api_data)

    @classmethod
    def retrieve_data(cls, id):
        return cls.client.get(f"/taskframes/{id}/").json()

    @classmethod
    def create(
        cls,
        data_type=None,
        task_type=None,
        json_schema=None,
        json_schema_url="",
        ui_schema=None,
        ui_schema_url="",
        instructions="",
        name="",
        review=True,
        redundancy=1,
        callback_url="",
        **kwargs,
    ):

        params = cls(
            data_type=data_type,
            task_type=task_type,
            json_schema=json_schema,
            json_schema_url=json_schema_url,
            ui_schema=ui_schema,
            ui_schema_url=ui_schema_url,
            instructions=instructions,
            name=name,
            review=review,
            redundancy=redundancy,
            callback_url=callback_url,
            **kwargs,
        ).to_dict()
        api_data = cls._create_from_dict(params)
        return cls.from_dict(api_data)

    @classmethod
    def update(
        cls,
        id,
        **kwargs,  # we don't specify kwargs to support partial updates and setting to None values.
    ):
        existing_instance = cls.retrieve(id)

        updatable_attrs = [
            "json_schema",
            "json_schema_url",
            "ui_schema",
            "ui_schema_url",
            "instructions",
            "name",
            "review",
            "redundancy",
            "callback_url",
        ]

        for kwarg, value in kwargs.items():
            if kwarg in updatable_attrs:
                setattr(existing_instance, kwarg, value)

        for kwarg, value in kwargs.items():
            if kwarg in cls.acceptable_params:
                existing_instance.kwargs[kwarg] = value

        params = existing_instance.to_dict()
        api_data = cls._update_from_dict(params)
        return cls.from_dict(api_data)

    def submit(self):
        if self.id:
            self._update_from_dict(self.to_dict())
        else:
            api_data = self._create_from_dict(self.to_dict())
            self.id = api_data["id"]
        if self.dataset is not None:
            self.dataset.submit(self.id)
        if self.trainingset is not None:
            self.trainingset.submit(self.id)
            self.submit_training_requirement(
                required_score=self.trainingset.required_score
            )
        if self.team:
            self.submit_team()

    @classmethod
    def _create_from_dict(cls, data):
        return cls.client.post("/taskframes/", json=data).json()

    @classmethod
    def _update_from_dict(cls, data):
        return cls.client.put(f"/taskframes/{data['id']}/", json=data).json()

    def preview(self):
        message = {
            "type": "set_preview",
            "data": {
                "taskframe": self.to_dict(),
            },
        }

        if self.dataset and len(self.dataset):
            item, custom_id, label, _id = self.dataset.get_random()
            serialized_item = self.dataset.serialize_item_preview(
                item, self.id, label=label
            )

            message["data"]["task"] = serialized_item

        css_id = str(int(random.random() * 10000))
        html = f"""
            <iframe id="frame_{css_id}" src="{APP_ENDPOINT}/embed/preview" frameBorder=0 style="width: 100%; height: 600px;"></iframe>
            <script>
            (function(){{
                var $iframe = document.querySelector('#frame_{css_id}');
                var init = false;
                postMessageHandler = function(e) {{
                    if (e.source !==  $iframe.contentWindow || e.data !== 'ready' || init) return;
                    $iframe.contentWindow.postMessage('{json.dumps(message)}', '*');
                    init = true;
                }}
                window.removeEventListener('message', postMessageHandler);
                window.addEventListener('message', postMessageHandler);
            }})()
            </script>
            """
        return display(HTML(html))

    def get_url(self):
        if not self.id:
            raise MissingId()
        return f"{APP_ENDPOINT}/taskframes/{self.id}"

    def open(self):
        display(Javascript(f'window.open("{self.get_url()}", "_blank");'))

    def progress(self):
        """Returns a dict of metrics related to the progress of the taskframe"""
        api_data = self.retrieve_data(self.id)

        return {
            "num_tasks": api_data.get("num_tasks"),
            "num_pending_work": api_data.get("num_pending_work"),
            "num_pending_review": api_data.get("num_pending_review"),
            "num_finished": api_data.get("num_pending_review"),
        }

    @classmethod
    def from_dict(cls, data):
        """Takes dict data from API, returns a Taskframe instance"""
        kwargs = data.get("params", {})

        return cls(
            id=data.get("id"),
            data_type=data.get("data_type"),
            task_type=data.get("task_type"),
            json_schema=data.get("json_schema"),
            json_schema_url=data.get("json_schema_url", ""),
            ui_schema=data.get("ui_schema"),
            ui_schema_url=data.get("ui_schema_url", ""),
            instructions=data.get("instructions", ""),
            name=data.get("name", ""),
            redundancy=data.get("redundancy"),
            review=data.get("review"),
            callback_url=data.get("callback_url", ""),
            **kwargs,
        )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "data_type": self.data_type,
            "task_type": self.task_type,
            "params": self._serialize_params(),
            "json_schema": self.json_schema,
            "json_schema_url": "",
            "ui_schema": self.ui_schema,
            "ui_schema_url": "",
            "instructions": self.instructions,
            "mode": "inhouse",
            "redundancy": self.redundancy,
            "review": self.review,
            "callback_url": self.callback_url,
        }

    def _serialize_params(self):
        return {
            k: self.kwargs.get(k) for k in self.acceptable_params if self.kwargs.get(k)
        }

    def _check_params(self, kwargs):

        invalid_params = []
        for kwarg, val in kwargs.items():
            if kwarg not in self.acceptable_params:
                invalid_params.append(kwarg)
        if invalid_params:
            raise InvalidParameter(f"invalid param(s): {', '.join(invalid_params)}")

    def fetch(self):
        warn("Deprecated, use cls.retrieve_data instead")
        response = self.client.get(f"/taskframes/{self.id}/")
        return response.json()

    # Export methods #########################

    def to_list(self):
        resp = self.client.get(
            f"/tasks/", params={"taskframe_id": self.id, "no_page": 1}
        )
        return resp.json()

    def to_dataframe(self):
        tasks = self.to_list()
        import pandas

        return pandas.DataFrame(tasks)

    def merge_to_dataframe(self, initial_dataframe, custom_id_column):
        exported_dataframe = self.to_dataframe()
        if "label" in initial_dataframe.columns:
            initial_dataframe = initial_dataframe.drop("label", axis=1)
        output_columns = list(initial_dataframe.columns) + ["label"]
        return initial_dataframe.merge(
            exported_dataframe, left_on=custom_id_column, right_on="custom_id"
        )[output_columns]

    def to_csv(self, path):
        tasks = self.to_list()
        if not tasks:
            raise ValueError("No data")
        keys = [
            "id",
            "custom_id",
            "taskframe_id",
            "input_data",
            "input_file",
            "input_url",
            "input_type",
            "status",
            "label",
            "initial_label",
            "priority",
        ]
        with open(path, "w") as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(tasks)

    def fetch_tasks(self):
        warn("Deprecated, use to_list instead")
        return self.to_list()

    # Dataset helper methods #########################

    def add_dataset_from_list(
        self, items, input_type=None, custom_ids=None, labels=None
    ):
        self.dataset = Dataset.from_list(
            items, input_type=input_type, custom_ids=custom_ids, labels=labels
        )

    def add_dataset_from_folder(
        self, path, custom_ids=None, labels=None, recursive=False, pattern="*"
    ):
        self.dataset = Dataset.from_folder(
            path,
            custom_ids=custom_ids,
            labels=labels,
            recursive=recursive,
            pattern=pattern,
        )

    def add_dataset_from_csv(
        self,
        csv_path,
        column=None,
        input_type=None,
        base_path=None,
        custom_id_column=None,
        label_column=None,
    ):
        self.dataset = Dataset.from_csv(
            csv_path,
            column=column,
            input_type=input_type,
            base_path=base_path,
            custom_id_column=custom_id_column,
            label_column=label_column,
        )

    def add_dataset_from_dataframe(
        self,
        dataframe,
        column=None,
        input_type=None,
        base_path=None,
        custom_id_column=None,
        label_column=None,
    ):
        self.dataset = Dataset.from_dataframe(
            dataframe,
            column=column,
            input_type=input_type,
            base_path=base_path,
            custom_id_column=custom_id_column,
            label_column=label_column,
        )

    def add_trainingset_from_list(
        self, items, input_type=None, custom_ids=None, labels=None, required_score=None
    ):
        self.trainingset = Trainingset.from_list(
            items,
            input_type=input_type,
            custom_ids=custom_ids,
            labels=labels,
            required_score=required_score,
        )

    def add_trainingset_from_folder(
        self,
        path,
        custom_ids=None,
        labels=None,
        recursive=False,
        pattern="*",
        required_score=None,
    ):
        self.trainingset = Trainingset.from_folder(
            path,
            custom_ids=custom_ids,
            labels=labels,
            recursive=recursive,
            pattern=pattern,
            required_score=required_score,
        )

    def add_trainingset_from_csv(
        self,
        csv_path,
        column=None,
        input_type=None,
        base_path=None,
        custom_id_column=None,
        label_column=None,
        required_score=None,
    ):
        self.trainingset = Trainingset.from_csv(
            csv_path,
            column=column,
            input_type=input_type,
            base_path=base_path,
            custom_id_column=custom_id_column,
            label_column=label_column,
            required_score=required_score,
        )

    def add_trainingset_from_dataframe(
        self,
        dataframe,
        column=None,
        input_type=None,
        base_path=None,
        custom_id_column=None,
        label_column=None,
        required_score=None,
    ):
        self.trainingset = Trainingset.from_dataframe(
            dataframe,
            column=column,
            input_type=input_type,
            base_path=base_path,
            custom_id_column=custom_id_column,
            label_column=label_column,
            required_score=required_score,
        )

    def submit_training_requirement(
        self,
        required_score=None,
    ):
        resp = self.client.post(
            f"/taskframes/{self.id}/set_training_requirement/",
            json={
                "required_score": required_score,
            },
        )

    # Team helper methods ###########################@

    def add_team(self, workers=[], reviewers=[], admins=[]):
        self.team = []
        workers = set(workers)
        reviewers = set(reviewers)
        admins = set(admins)
        if (
            workers.intersection(reviewers)
            or workers.intersection(admins)
            or reviewers.intersection(admins)
        ):
            raise ValueError("team members can't have multiple roles")

        team_data = []

        team_data.extend([{"role": "worker", "email": email} for email in workers])
        team_data.extend([{"role": "reviewer", "email": email} for email in reviewers])
        team_data.extend([{"role": "admin", "email": email} for email in admins])

        self.team = [TeamMember.from_dict(x) for x in team_data]

    def submit_team(self):
        existing_team = TeamMember.list(taskframe_id=self.id)
        for new_member in self.team:
            existing_member = _find_in_objects(existing_team, "email", new_member.email)
            if not existing_member:
                # create
                new_member.taskframe_id = self.id
                new_member.submit()
            elif (
                existing_member.role != new_member.role
                or existing_member.status != new_member.status
            ):
                resp = TeamMember.update(
                    taskframe_id=self.id,
                    id=existing_member.id,
                    role=new_member.role,
                    status=new_member.status,
                )

    def retrieve_team(self):
        return TeamMember.list(taskframe_id=self.id)


def _find_in_objects(items, key, value):
    try:
        return next(x for x in items if value in getattr(x, key, None) == value)
    except StopIteration:
        return None
