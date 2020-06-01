import json
import os
import random
from pathlib import Path

import requests
from IPython.display import HTML, Javascript, display

from .dataset import Dataset
from .utils import is_url

API_ENDPOINT = os.environ.get("TASKFRAME_API_ENDPOINT", "https://api.taskframe.ai")
API_VERSION = os.environ.get("TASKFRAME_API_VERSION", "v1")

API_URL = f"{API_ENDPOINT}/api/{API_VERSION}"


def open_file(*args, **kwargs):
    return open(*args, **kwargs)


class CustomIdsMismatch(Exception):
    def __init__(self, message="mismatch in length of dataset and custom_ids"):
        super().__init__(message)


class ApiError(Exception):
    """API responded with error"""

    def __init__(self, status_code, message):
        super().__init__("<Response [{}]> {}".format(status_code, message))
        self.status_code = status_code


class Taskframe(object):
    def __init__(
        self,
        data_type=None,
        task_type=None,
        classes=None,
        output_schema=None,
        instruction="",
        instruction_details=None,
        name=None,
        id=None,
    ):
        self.data_type = data_type
        self.task_type = task_type
        self.classes = classes
        self.output_schema = output_schema
        self.instruction = instruction
        self.instruction_details = instruction_details
        self.name = name
        self.id = id

        self.session = self.create_session()
        self.dataset_custom_ids = None

    def preview(self):
        message = {"type": "set_project", "data": self.to_dict()}
        css_id = str(int(random.random() * 10000))
        html = f"""
        <iframe id="frame_{css_id}" src="https://localhost:3000/embed/preview" frameBorder=0 style="width: 100%; height: 600px;"></iframe>
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

    def serialize_params(self):
        params = {}
        if self.classes:
            params["classes"] = self.classes
        return params

    def create_session(self):
        session = requests.Session()
        from . import api_key

        session.headers.update({"authorization": f"Token {api_key}"})
        return session

    def sync(self):
        response = self.session.get(f"{API_URL}/taskframes/{self.id}/",)
        return response.json()

    def submit(self):
        if self.id:
            self.update()
        else:
            self.create()

        if self.dataset is not None:
            self.submit_dataset()

        return self

    def update(self):
        self.session.put(f"{API_URL}/taskframes/{self.id}", json=self.to_dict())

    def create(self):
        self.session.post(f"{API_URL}/taskframes/", json=self.to_dict())

    def add_dataset_from_folder(self, path, custom_ids=None):
        self.dataset = Dataset.from_folder(path, custom_ids=custom_ids,)

        return self

    def add_dataset_from_list(self, items, input_type=None, custom_ids=None):
        self.dataset = Dataset.from_list(
            items, input_type=input_type, custom_ids=custom_ids
        )
        return self

    def add_dataset_from_csv(
        self,
        csv_path,
        column=None,
        input_type=None,
        base_path=None,
        custom_id_column=None,
    ):
        self.dataset = Dataset.from_csv(
            csv_path,
            column=column,
            input_type=input_type,
            base_path=base_path,
            custom_id_column=custom_id_column,
        )
        return self

    def add_dataset_from_dataframe(
        self,
        dataframe,
        column=None,
        input_type=None,
        base_path=None,
        custom_id_column=None,
    ):
        self.dataset = Dataset.from_dataframe(
            dataframe,
            column=column,
            input_type=input_type,
            base_path=base_path,
            custom_id_column=custom_id_column,
        )
        return self

    def submit_dataset(self):
        if self.dataset.input_type == Dataset.INPUT_TYPE_FILE:
            for item, custom_id in self.dataset.iterator():
                self.submit_dataset_file(item, custom_id=custom_id)
            return self

        # TODO: sub-batches.
        data = []
        for item, custom_id in self.dataset.iterator():
            data.append(
                remove_none_values(
                    {
                        "taskframe_id": self.id,
                        "custom_id": custom_id,
                        "input_data": (
                            item
                            if self.dataset.input_type == Dataset.INPUT_TYPE_DATA
                            else None
                        ),
                        "input_url": (
                            item
                            if self.dataset.input_type == Dataset.INPUT_TYPE_URL
                            else None
                        ),
                    }
                )
            )

        self.session.post(f"{API_URL}/tasks/", json=data)
        return self

    def submit_dataset_file(self, item, custom_id=None):
        path = Path(item)
        file_ = open_file(path, "rb")

        data = {
            "taskframe_id": (None, self.id),
            "input_file": (path.name, file_),
        }
        if custom_id:
            data["custom_id"] = (None, custom_id)
        resp = self.session.post(f"{API_URL}/tasks/", files=data)
        if resp.status_code >= 400:
            error_message = resp.text

            raise ApiError(resp.status_code, error_message)


def remove_none_values(obj):
    return {k: v for k, v in obj.items() if v is not None}
