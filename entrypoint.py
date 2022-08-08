import os
from src import main_routine
import warnings
warnings.filterwarnings("ignore")

def main():
    # log level
    os.environ["LOG_LEVEL"] = "DEBUG"


    problems_to_solve=[
        'adult'
    ]
    encoders_to_use = [
        #'OrdinalEncoder',
        #'OneHotEncoder',
        #'CatBoostEncoder',
        'CesamoEncoder'
    ]
    models_to_use=[
        'XGBClassifier',
        #'MLPClassifier',
    ]
    main_routine.start_benchmark(
        problems_to_solve,
        encoders_to_use,
        models_to_use
       )

if __name__ == '__main__':
    main()