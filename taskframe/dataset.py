import csv
from itertools import zip_longest
from pathlib import Path

from .utils import is_url


class CustomIdsMismatch(Exception):
    def __init__(self, message="mismatch in length of dataset and custom_ids"):
        super().__init__(message)


class Dataset(object):

    INPUT_TYPE_FILE = "file"
    INPUT_TYPE_URL = "url"
    INPUT_TYPE_DATA = "data"

    INPUT_TYPES = [INPUT_TYPE_FILE, INPUT_TYPE_URL, INPUT_TYPE_DATA]

    def __init__(self, items, input_type, custom_ids):
        self.items = items
        self.input_type = input_type
        self.custom_ids = custom_ids

    def iterator(self):
        if self.custom_ids:
            return zip(self.items, self.custom_ids)
        return zip_longest(self.items, [])  # zips with None

    @classmethod
    def from_list(cls, items, input_type=None, custom_ids=None):
        if custom_ids and len(custom_ids) != len(items):
            raise CustomIdsMismatch()
        return cls(
            items,
            input_type=input_type or guess_input_type(next(iter(items))),
            custom_ids=custom_ids,
        )

    @classmethod
    def from_folder(cls, path, custom_ids=None):
        return cls.from_list(
            list(Path(path).iterdir()),
            input_type=cls.INPUT_TYPE_FILE,
            custom_ids=custom_ids,
        )

    @classmethod
    def from_csv(
        cls,
        csv_path,
        column=None,
        input_type=None,
        base_path=None,
        custom_id_column=None,
    ):
        items = []

        custom_ids = [] if custom_id_column else None
        csv_path = Path(csv_path)
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            if not column:
                column = reader.fieldnames[0]
            base_path = Path(base_path) if base_path else csv_path.parents[0]
            first_item = next(iter(reader))[column]
            input_type = input_type or guess_input_type(first_item, base_path=base_path)

            needs_base_path_prepend = (
                True
                if input_type == cls.INPUT_TYPE_FILE
                and (base_path / Path(first_item)).exists()
                else False
            )
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            items = []
            for row in reader:
                if needs_base_path_prepend:
                    items.append(base_path / Path(row[column]))
                else:
                    items.append(row[column])

                if custom_id_column:
                    custom_ids.append(row[custom_id_column])
        return cls.from_list(items, input_type=input_type, custom_ids=custom_ids)

    @classmethod
    def from_dataframe(
        cls,
        dataframe,
        column=None,
        input_type=None,
        base_path=None,
        custom_id_column=None,
    ):
        base_path = Path(base_path) if base_path else Path()

        if not column:
            column = dataframe.columns[0]

        first_item = dataframe[column][0]

        input_type = input_type or guess_input_type(first_item, base_path)

        if input_type == cls.INPUT_TYPE_FILE:
            dataset = dataframe[column].apply(lambda x: Path(base_path) / Path(x))
        else:
            dataset = dataframe[column]
        custom_ids = None
        if custom_id_column:
            custom_ids = list(dataframe[custom_id_column])

        return cls(dataset, input_type=input_type, custom_ids=custom_ids)


def guess_input_type(first_item, base_path=Path()):
    if is_url(str(first_item)):
        return Dataset.INPUT_TYPE_URL
    if (isinstance(first_item, Path) or isinstance(first_item, str)) and (
        Path(first_item).exists() or (base_path / Path(first_item)).exists()
    ):
        return Dataset.INPUT_TYPE_FILE
    return Dataset.INPUT_TYPE_DATA
