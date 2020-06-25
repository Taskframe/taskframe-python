# Client Documentation

## Taskframe

class Taskframe

### __init__

```python
__init__(
    data_type=None,
    task_type=None,
    classes=None,
    output_schema=None,
    instruction="",
    instruction_details=None,
    name=None,
    id=None,
)
```

Parameters:

* `data_type`: Type of input data. Possible values : `image`, `text`, `html`, `pdf`.
* `task_type`: Type of predefined tasks (depending on the `data_type`):
  * `classification`: you may pass additional parameters:
    * `classes`: list of single-class options
    * `tags`: list of multi-class options
  * `text`: for transcription, text entry, sequence-to-sequence, etc. The annotator will have to fill a free text area.
  * `bounding_box`, `polygon`, `point`: for image annotations
  * `file_upload`: the annotator will have to upload a file(s) (experimental support for small files only). Extra parameters:
    * `multiple`: Boolean, whether multiple files are allowed
    * `files_accepted`: list of file extensions allowed, e.g. `[".jpg", ".png"]`. for all image formats simple pass `["image"]`
  * `custom`: Custom task based on a JSON Schema, Related parameters:
    * `output_schema`: a valid JSON Schema

  * `name`: the name that will appear in the platform list views (optional)
  * `instruction`: 1 line instruction that appears on top of the worker interface
  * `instruction_detail`: Free HTML section that will appear at the bottom of the worker interface. Allows safe HTML tags (`p`, `img`, etc.)
  * `id`: if you have already created your Taskframe you can simply create a Taskframe instance with the id, then call methods described below to fetch results, etc.

### add_dataset_from_list

```python
add_dataset_from_list(
        self, items, input_type=None, custom_ids=None, labels=None
    ):
```

Parameters:

* `items`: a list of items that will be annotated. items may be file paths, urls, or raw data (see below)
* `input_type` (optionnal): the type of items : `file`, `url`, `data`. If not provided it will be infered.
* `custom_ids` (optionnal): list of unique item ids. length should match `items`
* `labels` (optionnal): list of ground truth labels of your items (already known). length should match `items`. Items that already have a label will be treated as training data, those with a `None` value will have to be annotated by workers.

### add_dataset_from_folder

```python
add_dataset_from_folder(
        self, path, custom_ids=None, labels=None, recursive=False, pattern="*"
    )
```

Parameters:

* `path`: string or `Path` instance of the folder containing your files.
* `recursive`: Boolean. If true will also laod sub-directories.
* `pattern`: filter allowed file extensions.


### add_dataset_from_csv

```python
add_dataset_from_csv(
        self,
        csv_path,
        column=None,
        input_type=None,
        base_path=None,
        custom_id_column=None,
        label_column=None,
    )
```

Parameters:

* `csv_path`: string or `Path` containing the path to the CSV file.
* `column`: The column of the CSV containing your data. If undefined, takes the first column.
* `pattern`: filter allowed file extensions.
* `custom_id_column`: the column containing unique item ids.
* `label_column`: column containing ground truth labels of your items


### add_dataset_from_dataframe

```python
add_dataset_from_dataframe(
        self,
        dataframe,
        column=None,
        input_type=None,
        base_path=None,
        custom_id_column=None,
        label_column=None,
    )
```

Same `add_dataset_from_csv`

### add_team

```python
add_team(self, workers, reviewers=[]):
```

Specify team members that will be invited to collaborate. Parameters:

* `workers`: List of email addresses
* `reviewers`: List of email addresses

### submit

```python
submit(self)
```

This will perform the following actions:

Save the taskframe on the platform
Upload your dataset as new tasks
Upload your trainingset if specified, and configure the training required score
Send an email to team members to invite them to collaborate

### preview

```python
preview(self)
```

If you are in a Jupyter Notebook, you may call this method to display a preview of the worker interface.

### to_csv

```python
to_csv(self, path)
```

Export your results as a CSV

### to_dataframe

```python
to_dataframe(self)
```

Export your results data as a Pandas Dataframe in the return value.

### merge_to_dataframe

```python
merge_to_dataframe(self, dataframe, custom_id_column)
```

If your input Dataset is a Dataframe, you can merge your result labels as a new column in your original Dataframe.
Requires that you had submitted `custom_id` to be able to join rows.
Parameters:

* `dataframe`:  the dataframe to which we will add a new `labels` column
* `custom_id_colum`: the name of the column containing unique identifiers (should be the same as provided when dataset was initially submitted)