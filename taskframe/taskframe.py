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

API_ENDPOINT = os.environ.get("TASKFRAME_API_ENDPOINT", "https://api.taskframe.ai")

APP_ENDPOINT = os.environ.get("TASKFRAME_APP_ENDPOINT", "https://app.taskframe.ai")

API_VERSION = os.environ.get("TASKFRAME_API_VERSION", "v1")

API_URL = f"{API_ENDPOINT}/api/{API_VERSION}"


class CustomIdsMismatch(Exception):
    def __init__(self, message="mismatch in length of dataset and custom_ids"):
        super().__init__(message)


class Taskframe(object):
    def __init__(
        self,
        data_type=None,
        task_type=None,
        output_schema=None,
        instruction="",
        instruction_details=None,
        name=None,
        id=None,
        **kwargs,
    ):
        self.data_type = data_type
        self.task_type = task_type
        self.output_schema = output_schema
        self.instruction = instruction
        self.instruction_details = instruction_details
        self.name = name
        self.id = id
        self.client = Client()
        self.dataset = None
        self.trainingset = None
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
            "ui_schema": {},
            "ui_schema_url": "",
            "instruction": self.instruction,
            "instruction_details": self.instruction_details,
            "mode": "inhouse",
        }

    acceptable_task_params = ["classes", "tags", "multiple", "files_accepted"]

    def serialize_params(self):
        return {
            k: self.kwargs.get(k)
            for k in self.acceptable_task_params
            if self.kwargs.get(k)
        }

    def fetch(self):
        response = self.client.get(f"{API_URL}/taskframes/{self.id}/")
        return response.json()

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
            self.submit_dataset(self.dataset)
        if self.trainingset is not None:
            self.submit_dataset(self.trainingset)
            self.submit_training_requirement(
                required_score=self.trainingset.required_score
            )

    def update(self):
        self.client.put(f"{API_URL}/taskframes/{self.id}", json=self.to_dict())

    def create(self):
        self.client.post(f"{API_URL}/taskframes/", json=self.to_dict())

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

    def submit_dataset(self, dataset):
        # INPUT_TYPE_FILE doesnt support batches, post items one by one.
        if dataset.input_type == Dataset.INPUT_TYPE_FILE:
            for item, custom_id, label in dataset:
                data = dataset.serialize_item(
                    item, self.id, custom_id=custom_id, label=label
                )
                self.client.post(f"{API_URL}/tasks/", files=data)

            return

        # TODO: sub-batches.
        data = {
            "items": [
                dataset.serialize_item(item, self.id, custom_id=custom_id, label=label)
                for item, custom_id, label in dataset
            ]
        }
        resp = self.client.post(
            f"{API_URL}/tasks/", params={"taskframe_id": self.id}, json=data
        )

    def submit_training_requirement(
        self, required_score=0.9,
    ):
        resp = self.client.post(
            f"{API_URL}/taskframes/{self.id}/set_training_requirement/",
            data={"required_score": required_score,},
        )

    def fetch_tasks(self):
        resp = self.client.get(f"{API_URL}/tasks/?taskframe_id={self.id}&no_page=1",)
        return resp.json()

    def to_dataframe(self):
        tasks = self.fetch_tasks()
        import pandas

        return pandas.DataFrame(tasks)

    def merge_to_dataframe(self, dataframe, custom_id_column):
        answer_dataframe = self.to_dataframe()
        output_columns = list(dataframe.columns) + ["answer"]
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
            "answer",
        ]
        with open(path, "w") as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(tasks)

    def add_team(self, workers, reviewers=[]):
        self.team = []
        workers = set(workers)
        reviewers = set(reviewers)
        if workers.intersection(reviewers):
            raise ValueError("team members can't be both reviewer and worker")
        self.team.extend([{"role": "Worker", "email": x} for x in workers])
        self.team.extend([{"role": "Reviewer", "email": x} for x in reviewers])

    def submit_team(self):
        existing_team = self.fetch_team()
        existing_emails = [x["email"] for x in existing_team]
        for member in self.team:
            existing_member = find_in_dicts(existing_team, "email", member["email"])
            if not existing_member:
                resp = self.client.post(
                    f"{API_URL}/taskframes/{self.id}/users/", member
                )
                continue
            if existing_member["role"] != member["role"]:
                resp = self.client.put(
                    f"{API_URL}/taskframes/{self.id}/users/{existing_member['id']}/",
                    member,
                )

    def fetch_team(self):
        return self.client.get(
            f"{API_URL}/taskframes/{self.id}/users/?no_page=1"
        ).json()

    def preview(self):
        tf_message = {"type": "set_taskframe", "data": self.to_dict()}

        task_message = {}
        if self.dataset and len(self.dataset):
            item, _, _ = self.dataset.get_random()
            serialized_item = self.dataset.serialize_item_preview(item, self.id)

            task_message = json.dumps({"type": "set_task", "data": serialized_item})
        css_id = str(int(random.random() * 10000))
        html = f"""
            <iframe id="frame_{css_id}" src="{APP_ENDPOINT}/embed/preview" frameBorder=0 style="width: 100%; height: 600px;"></iframe>
            <script>
            (function(){{
                var $iframe = document.querySelector('#frame_{css_id}');
                var init = false;
                postMessageHandler = function(e) {{
                    if (e.source !==  $iframe.contentWindow || e.data !== 'ready' || init) return;
                    $iframe.contentWindow.postMessage('{json.dumps(tf_message)}', '*');
                    $iframe.contentWindow.postMessage('{task_message}', '*');
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
