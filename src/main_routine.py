# set log
from nis import cat
from config.logging import set_logger
log=set_logger()
# third-part-modules
import os
# inner-modules
from src import data_manager
from src import cat_encoders
from src import model_manager
from config import general as general_config


def start_benchmark(problems_to_solve, encoders_to_use, models_to_use):

    # array 'name' problems to solve
    log.info(f"datasets to solve: {problems_to_solve}")
    log.info(f"encoders to use: {encoders_to_use}")
    log.info(f"models to use: {models_to_use}")

    for dataset_name in problems_to_solve:
        # Set
        os.environ["DATASET_NAME"] = str(dataset_name)

        # load dataset config
        metadata_dataset_path = f'./dataset_metadata/{dataset_name}.json'
        metadata_dataset = data_manager.load_json(metadata_dataset_path)
        cat_cols=metadata_dataset["cat_cols"]
        # load dataset in a dataframe
        raw_df = data_manager.load_csv(metadata_dataset["relative_path_to_dataset"])
        print(raw_df.info())
        # clean_data
        df = data_manager.clean_data(raw_df, metadata_dataset)
        # split features and target
        features_data, target = data_manager.split_features_and_target(df)
        # estimate the number of columns to add for the case of based on OHE encoders
        cols_to_add_lc = sum([df[x].nunique() for x in cat_cols])
        log.info(f'estimated added cols for OHE-based methods: {cols_to_add_lc}')
        counts = df['target'].value_counts().tolist()
        log.info(f"class distribution, positive %: {counts[1]/counts[0]}")
        log.info(f"class distribution, negative % {1-(counts[1]/counts[0])}")

        # encoders array
        encoders = cat_encoders.set(cat_cols, encoders_to_use)

        # set models
        models = model_manager.set(models_to_use)
        #print(models)
        # solve the problem with each different model
        for model in models:
            # apply encoder
            for encoder in encoders:
                log.info(f'Solving classification problem: {dataset_name}, with model: {model["model_name"]} and encoder: {encoder["encoder_name"]}')
                f1_score, elapsed_time = model_manager.solve_class_problem(features_data, target, 3, model, encoder)
                log.info(f'f1_score achieved: {f1_score}')
                data_manager.save_results(
                    dataset_name, 
                    model["model_name"], 
                    encoder["encoder_name"], 
                    f1_score, 
                    elapsed_time,
                    general_config.path_to_performance_results    
                )

        # models

