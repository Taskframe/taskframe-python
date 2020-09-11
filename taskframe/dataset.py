import base64
import csv
import json
import mimetypes
import random
from datetime import datetime
from pathlib import Path

from .client import Client
from .utils import is_url, remove_empty_values

mimetypes.init()


class InvalidData(Exception):
    pass


def get_or_none(list_, idx):
    return list_[idx] if idx < len(list_) else None


def open_file(*args, **kwargs):  # for easier mocked unit_tests
    return open(*args, **kwargs)


def guess_input_type(first_item, base_path=Path()):
    if is_url(str(first_item)):
        return Dataset.INPUT_TYPE_URL
    try:
        if (isinstance(first_item, Path) or isinstance(first_item, str)) and (
            Path(first_item).exists() or (base_path / Path(first_item)).exists()
        ):
            return Dataset.INPUT_TYPE_FILE
    except OSError as exc:
        if exc.errno == 36:  # Filename too long: its probably raw data.
            pass
        else:
            raise
    return Dataset.INPUT_TYPE_DATA


class CustomIdsLengthMismatch(Exception):
    def __init__(self, message="mismatch in length of dataset and custom_ids"):
        super().__init__(message)


class LabelsLengthMismatch(Exception):
    def __init__(self, message="mismatch in length of dataset and labels"):
        super().__init__(message)


class MissingLabelsMismatch(Exception):
    def __init__(self, message="All labels should be defined"):
        super().__init__(message)


class Dataset(object):

    INPUT_TYPE_FILE = "file"
    INPUT_TYPE_URL = "url"
    INPUT_TYPE_DATA = "data"

    INPUT_TYPES = [INPUT_TYPE_FILE, INPUT_TYPE_URL, INPUT_TYPE_DATA]

    client = Client()

    def __init__(self, items, ids=None, custom_ids=None, labels=None, **kwargs):

        self.items = self.prepare_items(items, **kwargs)

        self.sanity_check(self.items, custom_ids, labels)

        for item in self.items:
            self.sanity_check_item(item)

        self.custom_ids = custom_ids or []
        self.labels = labels or []
        self.ids = ids or []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if i < 0 or i >= len(self.items):
            raise IndexError()
        return (
            self.items[i],
            get_or_none(self.custom_ids, i),
            get_or_none(self.labels, i),
            get_or_none(self.ids, i),
        )

    def serialized_items(self, taskframe_id):
        for item, custom_id, label, _id in self:
            yield self.serialize_item(
                item, taskframe_id, custom_id=custom_id, label=label
            )

    def get_random(self):
        idx = random.randint(0, len(self) - 1)
        return self[idx]

    @classmethod
    def get_dataset_class(cls, input_type):
        dataset_class_map = {
            "url": UrlDataset,
            "file": FileDataset,
            "data": DataDataset,
        }
        if input_type not in dataset_class_map.keys():
            raise ValueError(
                f'input type should be in {", ".join(dataset_class_map.keys())}'
            )
        return dataset_class_map[input_type]

    @classmethod
    def from_list(
        cls, items, input_type=None, custom_ids=None, labels=None, base_path=None
    ):

        input_type = input_type or guess_input_type(next(iter(items)))

        return cls.get_dataset_class(input_type)(
            items, custom_ids=custom_ids, labels=labels, base_path=base_path
        )

    @classmethod
    def from_folder(
        cls, path, custom_ids=None, labels=None, recursive=False, pattern="*"
    ):
        items = []
        path = Path(path)
        if recursive:
            items = path.rglob(pattern)
        else:
            items = path.glob(pattern)

        items = [x for x in items if x.is_file() and x.name != ".DS_Store"]

        return cls.from_list(
            items, input_type=cls.INPUT_TYPE_FILE, custom_ids=custom_ids, labels=labels,
        )

    @classmethod
    def from_csv(
        cls,
        csv_path,
        column=None,
        input_type=None,
        base_path=None,
        custom_id_column=None,
        label_column=None,
    ):
        items = []

        custom_ids = [] if custom_id_column else None
        labels = [] if label_column else None
        csv_path = Path(csv_path)
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            if not column:
                column = reader.fieldnames[0]
            base_path = Path(base_path) if base_path else csv_path.parents[0]
            first_item = next(iter(reader))[column]
            input_type = input_type or guess_input_type(first_item, base_path=base_path)

        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            items = []
            for row in reader:
                items.append(row[column])
                if custom_id_column:
                    custom_ids.append(row[custom_id_column])
                if label_column:
                    labels.append(row[label_column])
        return cls.from_list(
            items,
            input_type=input_type,
            custom_ids=custom_ids,
            labels=labels,
            base_path=base_path,
        )

    @classmethod
    def from_dataframe(
        cls,
        dataframe,
        column=None,
        input_type=None,
        base_path=None,
        custom_id_column=None,
        label_column=None,
    ):
        base_path = Path(base_path) if base_path else Path()
        dataframe = dataframe.fillna("")

        if not column:
            column = dataframe.columns[0]

        first_item = dataframe[column][0]
        input_type = input_type or guess_input_type(first_item, base_path)

        dataset = dataframe[column]

        custom_ids = []
        labels = []
        if custom_id_column:
            custom_ids = list(dataframe[custom_id_column])

        if label_column:
            labels = list(dataframe[label_column])

        return cls.get_dataset_class(input_type)(
            dataset, custom_ids=custom_ids, labels=labels, base_path=base_path
        )

    def serialize_item(
        self, item, taskframe_id, custom_id=None, label=None, base64enc=False
    ):
        raise NotImplementedError()

    def sanity_check(self, items, custom_ids, labels):
        if custom_ids and len(custom_ids) != len(items):
            raise CustomIdsLengthMismatch()

        if labels and len(labels) != len(items):
            raise LabelsLengthMismatch()

    def prepare_items(self, items, **kwargs):
        return items

    def sanity_check_item(self, item):
        raise NotImplementedError()

    def serialize_item_preview(self, *args, **kwargs):
        return self.serialize_item(*args, **kwargs)

    def submit(self, taskframe_id):
        data = {"items": list(self.serialized_items(taskframe_id))}
        resp = self.client.post(
            f"/tasks/", params={"taskframe_id": taskframe_id}, json=data
        )

        resp_data = resp.json()

        self.ids = [x["id"] for x in resp_data]
        return


class FileDataset(Dataset):

    input_type = "file"
    max_file_size = 50 * 1000 * 1000  # 50MB

    def prepare_items(self, items, base_path=None):
        base_path = Path(base_path) if base_path else None
        needs_preprend = base_path and (base_path / Path(items[0])).exists()
        if needs_preprend:
            return [base_path / Path(item) for item in items]
        return items

    def sanity_check_item(self, item):
        item = Path(item)
        if not item.exists():
            raise InvalidData(f"file does not exist: {str(item)}")
        if Path(item).stat().st_size > self.max_file_size:
            raise InvalidData(f"File larger than 50MB: {str(item)}")
        # TODO: check that item matches input_type.

    def serialize_item(self, item, taskframe_id, custom_id=None, label=None):
        path = Path(item)
        file_ = open_file(path, "rb")
        data = {
            "taskframe_id": (None, taskframe_id),
            "input_file": (path.name, file_),
            "input_type": (None, self.input_type),
        }
        if custom_id:
            data["custom_id"] = (None, custom_id)
        if label:
            data["initial_label"] = (None, json.dumps(label))

        return data

    def serialize_item_preview(self, item, taskframe_id, custom_id=None, label=None):
        """In preview, files are base64 encoded and passed as data urls."""
        path = Path(item)
        mimetype = mimetypes.types_map[path.suffix]
        file_ = open(path, "rb")
        contents = file_.read()
        data_url = f"data:{mimetype};base64,{base64.b64encode(contents).decode()}"
        return remove_empty_values(
            {
                "custom_id": custom_id,
                "input_url": data_url,
                "input_type": "url",
                "initial_label": label,
                "taskframe_id": taskframe_id,
            }
        )

    def submit(self, taskframe_id):
        # INPUT_TYPE_FILE doesnt support batches, post items one by one.
        resp_data = []
        for data in self.serialized_items(taskframe_id):
            resp = self.client.post(f"/tasks/", files=data)
            resp_data.append(resp.json())
        self.ids = [x["id"] for x in resp_data]
        return


class UrlDataset(Dataset):

    input_type = "url"

    def serialize_item(self, item, taskframe_id, custom_id=None, label=None):
        return remove_empty_values(
            {
                "custom_id": custom_id,
                "input_url": item,
                "input_type": self.input_type,
                "initial_label": label,
                "taskframe_id": taskframe_id,
            }
        )

    def sanity_check_item(self, item):
        if not is_url(item):
            raise InvalidData("Not a URL: {item}")
        # TODO: check that item matches input_type.


class DataDataset(Dataset):

    input_type = "data"

    def serialize_item(self, item, taskframe_id, custom_id=None, label=None):
        return remove_empty_values(
            {
                "custom_id": custom_id,
                "input_data": item,
                "input_type": self.input_type,
                "initial_label": label,
                "taskframe_id": taskframe_id,
            }
        )

    def sanity_check_item(self, item):
        pass  # TODO: check that item matches input_type.


class TrainingsetMixin(object):
    is_training = True

    def sanity_check(self, items, custom_ids, labels):
        super().sanity_check(items, custom_ids, labels)

        if not all(labels):
            raise MissingLabelsMismatch()

    def serialize_item(self, item, taskframe_id, custom_id=None, label=None):
        resp = super().serialize_item(
            item, taskframe_id, custom_id=custom_id, label=label
        )
        if self.input_type == "file":
            resp["is_training"] = (None, True)
        else:
            resp["is_training"] = True
        return resp


class UrlTrainingSet(TrainingsetMixin, UrlDataset):
    pass


class FileTrainingSet(TrainingsetMixin, FileDataset):
    pass


class DataTrainingSet(TrainingsetMixin, DataDataset):
    pass


class Trainingset(Dataset):
    @classmethod
    def get_dataset_class(cls, input_type):
        class_map = {
            "url": UrlTrainingSet,
            "file": FileTrainingSet,
            "data": DataTrainingSet,
        }
        if input_type not in class_map.keys():
            raise ValueError(f'input type should be in {", ".join(class_map.keys())}')
        return class_map[input_type]

    @classmethod
    def from_list(cls, *args, required_score=None, **kwargs):
        instance = super().from_list(*args, **kwargs)
        instance.required_score = required_score
        return instance

    @classmethod
    def from_folder(cls, *args, required_score=None, **kwargs):
        instance = super().from_folder(*args, **kwargs)
        instance.required_score = required_score
        return instance

    @classmethod
    def from_csv(cls, *args, required_score=None, **kwargs):
        instance = super().from_csv(*args, **kwargs)
        instance.required_score = required_score
        return instance

    @classmethod
    def from_dataframe(cls, *args, required_score=None, **kwargs):
        instance = super().from_dataframe(*args, **kwargs)
        instance.required_score = required_score
        return instance
