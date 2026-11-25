# Configuration

Copy the example YAML, then set SMARTX_DATA_ROOT and either SMARTX_CHECKPOINT
or the model checkpoint field.

Each manifest row represents one case. The image_paths column accepts a JSON
list or paths separated by semicolons or pipes. Labels may be stored in one
labels column or in separate columns selected with dataset.label_columns.
