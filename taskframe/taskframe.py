import csv
import json
import os
import random
from pathlib import Path

import requests
from IPython.display import HTML, Javascript, display

from .client import Client
from .dataset import Dataset, Trainingset
from .utils import is_url

APP_ENDPOINT = os.environ.get("TASKFRAME_APP_ENDPOINT", "https://app.taskframe.ai")


class CustomIdsMismatch(Exception):
    def __init__(self, message="mismatch in length of dataset and custom_ids"):
        super().__init__(message)


class Taskframe(object):
    def __init__(
        self,
        data_type=None,
        task_type=None,
        output_schema=None,
        ui_schema=None,
        instruction="",
        instruction_details="",
        name="",
        id=None,
        review=True,
        redundancy=1,
        **kwargs,
    ):
        self.data_type = data_type
        self.task_type = task_type
        self.output_schema = output_schema
        self.ui_schema = ui_schema
        self.instruction = instruction
        self.instruction_details = instruction_details
        self.name = name
        self.id = id
        self.client = Client()
        self.dataset = None
        self.trainingset = None
        self.review = review
        self.redundancy = redundancy
        self.workers = []
        self.reviewers = []
        self.kwargs = kwargs

    def to_dict(self):
        return {
            "name": self.name,
            "data_type": self.data_type,
            "task_type": self.task_type,
            "params": self.serialize_params(),
            "output_schema": self.output_schema,
            "output_schema_url": "",
            "ui_schema": self.ui_schema,
            "ui_schema_url": "",
            "instruction": self.instruction,
            "instruction_details": self.instruction_details,
            "mode": "inhouse",
            "redundancy": self.redundancy,
            "requires_review": self.review,
        }

    acceptable_task_params = [
        "classes",
        "tags",
        "image_classes",
        "image_tags",
        "region_classes",
        "region_tags",
        "multiple",
        "files_accepted",
    ]

    def serialize_params(self):
        return {
            k: self.kwargs.get(k)
            for k in self.acceptable_task_params
            if self.kwargs.get(k)
        }

    def fetch(self):
        response = self.client.get(f"/taskframes/{self.id}/")
        return response.json()

    @classmethod
    def retrieve(cls, id):
        client = Client()
        data = client.get(f"/taskframes/{id}/").json()
        return cls(
            data_type=data["data_type"],
            task_type=data["task_type"],
            output_schema=data["output_schema"],
            ui_schema=data["ui_schema"],
            instruction=data["instruction"],
            instruction_details=data["instruction_details"],
            name=data["name"],
            id=id,
            classes=data["params"].get("classes"),
            tags=data["params"].get("tags"),
            image_classes=data["params"].get("image_classes"),
            image_tags=data["params"].get("image_tags"),
            region_classes=data["params"].get("region_classes"),
            region_tags=data["params"].get("region_tags"),
            multiple=data["params"].get("multiple"),
            files_accepted=data["params"].get("files_accepted"),
            redundancy=data["redundancy"],
            review=data["requires_review"],
        )

    def progress(self):
        data = self.fetch()

        return {
            "num_tasks": data.get("num_tasks"),
            "num_pending_work": data.get("num_pending_work"),
            "num_pending_review": data.get("num_pending_review"),
            "num_finished": data.get("num_pending_review"),
        }

    def submit(self):
        if self.id:
            self.update()
        else:
            self.create()
        if self.dataset is not None:
            self.dataset.submit(self.id)
        if self.trainingset is not None:
            self.trainingset.submit(self.id)
            self.submit_training_requirement(
                required_score=self.trainingset.required_score
            )

    def update(self):
        self.client.put(f"/taskframes/{self.id}/", json=self.to_dict())

    def create(self):
        resp = self.client.post(f"/taskframes/", json=self.to_dict())
        self.id = resp.json()["id"]
        print(f"created taskframe of id: {self.id}")

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
        self, required_score=None,
    ):
        resp = self.client.post(
            f"/taskframes/{self.id}/set_training_requirement/",
            data={"required_score": required_score,},
        )

    def fetch_tasks(self):
        resp = self.client.get(f"/tasks/?taskframe_id={self.id}&no_page=1",)
        return resp.json()

    def to_dataframe(self):
        tasks = self.fetch_tasks()
        import pandas

        return pandas.DataFrame(tasks)

    def merge_to_dataframe(self, dataframe, custom_id_column):
        answer_dataframe = self.to_dataframe()
        output_columns = list(dataframe.columns) + ["label"]
        return dataframe.merge(
            answer_dataframe, left_on=custom_id_column, right_on="custom_id"
        )[output_columns]

    def to_csv(self, path):
        tasks = self.fetch_tasks()
        if not tasks:
            raise ValueError("No data")
        keys = [
            "id",
            "custom_id",
            "taskframe_id",
            "taskframe_name",
            "input_data",
            "input_file",
            "input_url",
            "input_type",
            "status",
            "label",
        ]
        with open(path, "w") as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(tasks)

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
        self.team.extend(
            [{"role": "Worker", "email": x, "status": "active"} for x in workers]
        )
        self.team.extend(
            [{"role": "Reviewer", "email": x, "status": "active"} for x in reviewers]
        )
        self.team.extend(
            [{"role": "Admin", "email": x, "status": "active"} for x in admins]
        )

    def submit_team(self):
        existing_team = self.fetch_team()
        existing_emails = [x["email"] for x in existing_team]
        for member in self.team:
            existing_member = find_in_dicts(existing_team, "email", member["email"])
            if not existing_member:
                resp = self.client.post(f"/taskframes/{self.id}/users/", member)
                continue
            if existing_member["role"] != member["role"]:
                resp = self.client.put(
                    f"/taskframes/{self.id}/users/{existing_member['id']}/", member,
                )

    def fetch_team(self):
        return self.client.get(f"/taskframes/{self.id}/users/?no_page=1").json()

    def preview(self):
        message = {"type": "set_preview", "data": {"taskframe": self.to_dict(),}}

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


def find_in_dicts(items, key, value):
    try:
        return next(x for x in items if value in x.get(key) == value)
    except StopIteration:
        return None
