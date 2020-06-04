import base64
import csv
import mimetypes
import random
from pathlib import Path

from .utils import is_url, remove_none_values

mimetypes.init()


def get_or_none(list_, idx):
    return list_[idx] if idx < len(list_) else None


def open_file(*args, **kwargs):
    return open(*args, **kwargs)


def guess_input_type(first_item, base_path=Path()):
    if is_url(str(first_item)):
        return Dataset.INPUT_TYPE_URL
    if (isinstance(first_item, Path) or isinstance(first_item, str)) and (
        Path(first_item).exists() or (base_path / Path(first_item)).exists()
    ):
        return Dataset.INPUT_TYPE_FILE
    return Dataset.INPUT_TYPE_DATA


class CustomIdsLengthMismatch(Exception):
    def __init__(self, message="mismatch in length of dataset and custom_ids"):
        super().__init__(message)


class LabelsLengthMismatch(Exception):
    def __init__(self, message="mismatch in length of dataset and custom_ids"):
        super().__init__(message)


class Dataset(object):

    INPUT_TYPE_FILE = "file"
    INPUT_TYPE_URL = "url"
    INPUT_TYPE_DATA = "data"

    INPUT_TYPES = [INPUT_TYPE_FILE, INPUT_TYPE_URL, INPUT_TYPE_DATA]

    def __init__(self, items, custom_ids=None, labels=None, **kwargs):
        self.items = items
        self.custom_ids = custom_ids or []
        self.labels = labels or []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if i < 0 or i >= len(self.items):
            raise IndexError()
        return (
            self.items[i],
            get_or_none(self.custom_ids, i),
            get_or_none(self.labels, i),
        )

    def get_random(self):
        idx = random.randint(0, len(self) - 1)
        return self[idx]

    @classmethod
    def from_list(
        cls, items, input_type=None, custom_ids=None, labels=None, base_path=None
    ):
        if custom_ids and len(custom_ids) != len(items):
            raise CustomIdsLengthMismatch()

        if labels and len(labels) != len(items):
            raise LabelsLengthMismatch()

        input_type = input_type or guess_input_type(next(iter(items)))

        return cls.get_dataset_class(input_type)(
            items, custom_ids=custom_ids, labels=labels, base_path=base_path
        )

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
    def from_folder(cls, path, custom_ids=None, labels=None):
        return cls.from_list(
            list(Path(path).iterdir()),
            input_type=cls.INPUT_TYPE_FILE,
            custom_ids=custom_ids,
            labels=labels,
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

    def serialize_item_preview(self, *args, **kwargs):
        return self.serialize_item(*args, **kwargs)


class FileDataset(Dataset):

    input_type = "file"

    def __init__(self, *args, base_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        base_path = Path(base_path) if base_path else None
        needs_preprend = base_path and (base_path / Path(self.items[0])).exists()

        if needs_preprend:
            self.items = [base_path / Path(item) for item in self.items]

    def serialize_item(self, item, taskframe_id, custom_id=None, label=None):
        path = Path(item)
        file_ = open_file(path, "rb")
        data = {
            "taskframe_id": (None, taskframe_id),
            "input_file": (path.name, file_),
            "input_type": self.input_type,
            "is_training": (None, bool(label)),
        }
        if custom_id:
            data["custom_id"] = (None, custom_id)
        if label:
            data["answer"] = (None, label)

        return data

    def serialize_item_preview(self, item, taskframe_id, custom_id=None, label=None):
        """In preview, files are base64 encoded and passed as data urls."""
        path = Path(item)
        mimetype = mimetypes.types_map[path.suffix]
        file_ = open(path, "rb")
        contents = file_.read()
        data_url = f"data:{mimetype};base64,{base64.b64encode(contents).decode()}"
        return remove_none_values(
            {
                "custom_id": custom_id if custom_id else None,
                "input_url": data_url,
                "input_type": "url",
                "answer": label if label else None,
                "is_training": bool(label),
                "taskframe_id": taskframe_id,
            }
        )


class UrlDataset(Dataset):

    input_type = "url"

    def serialize_item(self, item, taskframe_id, custom_id=None, label=None):
        return remove_none_values(
            {
                "custom_id": custom_id if custom_id else None,
                "input_url": item,
                "input_type": self.input_type,
                "answer": label if label else None,
                "is_training": bool(label),
                "taskframe_id": taskframe_id,
            }
        )


class DataDataset(Dataset):

    input_type = "data"

    def serialize_item(self, item, taskframe_id, custom_id=None, label=None):
        return remove_none_values(
            {
                "custom_id": custom_id if custom_id else None,
                "input_data": item,
                "input_type": self.input_type,
                "answer": label if label else None,
                "is_training": bool(label),
                "taskframe_id": taskframe_id,
            }
        )
