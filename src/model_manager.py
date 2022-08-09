# set log
from nis import cat
from config.logging import set_logger
log=set_logger()
# third-part-modules
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from statistics import mean
import time
# models
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def set(models_list):
    all_models_available = [
        {
            "model_name":"XGBClassifier",
            "model_obj":XGBClassifier()
        },
        {
            "model_name":"MLPClassifier",
            "model_obj":MLPClassifier(solver='adam', activation="relu",hidden_layer_sizes=(25, 25, 25), random_state=1)
        },
        {
            "model_name":"GaussianNB",
            "model_obj":GaussianNB()
        },
        {
            "model_name":"LogisticRegression",
            "model_obj":LogisticRegression()
        },
        {
            "model_name":"SVC",
            "model_obj":SVC()
        },    
        {
            "model_name":"RandomForestClassifier",
            "model_obj":RandomForestClassifier()
        }

    ]

    final_models=[]    
    for model in all_models_available:
        if model["model_name"] in models_list:
            final_models.append(model)

    return final_models

def solve_class_problem(data_features, data_target, cv, model, encoder_data):
    #X_train, x_Test, y_train, y_Test = train_test_split(data_features, data_target, test_size = 0.2, random_state = 43)
    skfolds = StratifiedKFold(n_splits=cv, random_state=42, shuffle=True)
    fold_counter = 1
    f1_score_results=[]
    elapsed_times=[]
    for train_index, test_index in skfolds.split(data_features, data_target):
        log.info(f"Applying fold number {fold_counter} out of {cv}")
        X_train_fold = data_features.iloc[train_index]
        y_train_fold = data_target[train_index]     
        X_test_fold = data_features.iloc[test_index]     
        y_test_fold = data_target[test_index]
        start_timer = time.time()
        
        # apply categorical encoder
        encoder_obj = encoder_data['encoder_obj']
        log.info(f'Fitting the encoder')
        encoder_obj = encoder_obj.fit(X_train_fold, y_train_fold)
        log.info(f'Transforming data for X_train_fold by the encoder')
        X_train_fold = encoder_obj.transform(X_train_fold, y_train_fold)
        X_train_fold=X_train_fold.fillna(X_train_fold.mean())
        
        # train model
        model_obj = model['model_obj']
        log.info(f'Fitting the model')
        model_obj.fit(X_train_fold, y_train_fold)
        
        # apply encoder transform to X Test
        log.info("Transforming data for X_test_fold by the encoder")
        X_test_fold = encoder_obj.transform(X_test_fold)
        X_test_fold=X_test_fold.fillna(X_test_fold.mean())
        

        y_pred = model_obj.predict(X_test_fold) 
        end_timer = time.time()
        elapsed_time = end_timer - start_timer
        f1_score_results.append(f1_score(y_pred, y_test_fold))
        elapsed_times.append(elapsed_time)
        # increment counter
        fold_counter+=1

    f1_score_results_mean = mean(f1_score_results)
    avg_elapsed_time = mean(elapsed_times)
    
    return f1_score_results_mean, avg_elapsed_time