# set log
from nis import cat
from config.logging import set_logger
log=set_logger()
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from time import gmtime, strftime
from datetime import datetime


def load_json(path_and_name):
    with open(path_and_name, "r") as read_file:
        return json.load(read_file)

def load_csv(path_and_name):
    return pd.read_csv(path_and_name)

def clean_data(df, dataset_config):
    # remove unnamed cols
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # set target
    df["target"] = df[dataset_config["target"]] == dataset_config['positive_values_are_represented_by']
    df.drop([dataset_config["target"]] + dataset_config["time_cols"] + dataset_config["id_cols"]+ dataset_config["cols_to_delete"], axis=1, inplace=True)
    # remove nan values
    df.dropna(axis=0, inplace=True)
    # remove empty strings or only spaces
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # scale num cols
    print(df[dataset_config['num_cols']].head(10))
    mms = MinMaxScaler()
    df[dataset_config['num_cols']] = mms.fit_transform(df[dataset_config['num_cols']])
    return df

def split_features_and_target(data):
    data_features=data.drop(["target"], axis=1)
    data_target = data["target"].to_numpy()
    return data_features, data_target 

def save_results(dataset_name, model_name, encoder_name, f1_score, elapsed_time, path_to_dir):

    path = os.path.join(path_to_dir, dataset_name)
    # Check whether the specified path exists or not
    if not os.path.exists(path):
        # Create a new directory because it does not exist 
        os.makedirs(path)
    formated_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result = {
        "dataset_name":[dataset_name],
        "model_name":[model_name],
        "encoder_name":[encoder_name],
        "f1_score":[f1_score],
        "elapsed_time":[elapsed_time],
        "exec_date":[formated_date]
    } 
   
    #current_date=strftime("%Y%m%d%H%M%S", gmtime())
    current_date=datetime.now().strftime("%Y%m%d%H%M%S")
    df = pd.DataFrame(result)
    # save dataframe
    result_file_name = f'{current_date}_{model_name}_{encoder_name}.csv'
    final_path_and_name=os.path.join(path, result_file_name)
    df.to_csv(final_path_and_name, index=False)
    log.info(f'Resuls saved in {final_path_and_name}')