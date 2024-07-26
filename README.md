This is the repository with the config templates for the [**sequifier**](https://github.com/0xideas/sequifier) package.

The simplest workflow is the following:

1. Run `pip install sequifier`
2. Clone this repository to your project name:
    `git clone git@github.com:0xideas/sequifier-config.git YOUR PROJECT NAME`
3. Create a folder `data` and copy your data to that folder
4. Ensure data quality, such as the removal of outliers and no NaN values in the relevant columns, and that there are two columns `sequenceId` and `itemPosition`, which indicate which sequence a row belongs to and which position in the sequence the row has, respectively
5. Adapt the fields `data_path`, `selected_columns` and `target_columns` in `preprocess.yaml`
6. Run `sequifier preprocess` from the project root
7. Adapt the fields `ddconfig_path` to `configs/ddconfigs/DATASET-NAME.json`, `selected_columns`, `target_columns`, `target_column_types` and `criterion` in `train.yaml`
8. Run `sequifier train` from the project root to train the model
9. Adapt the fields `ddconfig_path`, `model_path`, `data_path`, `selected_columns`, `target_columns` and `target_column_types` in infer.yaml
10. Run `sequifier infer`

Voila, you have extrapolated your categorical or real series using your custom transformer model!