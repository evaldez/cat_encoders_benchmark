import os
from src import main_routine
import warnings
warnings.filterwarnings("ignore")

def main():
    # log level
    # os.environ["LOG_LEVEL"] = "INFO"

    problems_to_solve=[
        #'adult',
        'cat_in_the_data'
    ]
    encoders_to_use = [
        'OrdinalEncoder',
        #'OneHotEncoder',
        'CatBoostEncoder',
        #'CesamoEncoder',
        'RandomEncoder',
        'TargetEncoder'
    ]
    models_to_use=[
        #'XGBClassifier',
        #'MLPClassifier',
        #'GaussianNB',
        'LogisticRegression',
        #'SVC',
        'RandomForestClassifier'

    ]
    main_routine.start_benchmark(
            problems_to_solve,
            encoders_to_use,
            models_to_use
       )

if __name__ == '__main__':
    main()