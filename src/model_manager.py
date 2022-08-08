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



def set(models_list):
    all_models_available = [
        {
            "model_name":"XGBClassifier",
            "model_obj":XGBClassifier()
        },
        {
            "model_name":"MLPClassifier",
            "model_obj":MLPClassifier(solver='adam', activation="relu", alpha=1e-5, max_iter=2000,hidden_layer_sizes=(64, 64, 64), random_state=1)
        },


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
        encoder_obj = encoder_obj.fit(X_train_fold, y_train_fold)
        #print("---------- before X_train_fold encoded ----------")
        #print(X_train_fold.head(10))
        X_train_fold = encoder_obj.transform(X_train_fold, y_train_fold)
        #print("---------- after X_train_fold encoded ----------")
        #print(X_train_fold.head(10))
        
        # train model
        model_obj = model['model_obj']
        model_obj.fit(X_train_fold, y_train_fold)
        
        # apply encoder transform to X Test
        #print("---------- before X_test_fold encoded ----------")
        #print(X_test_fold.head(10))
        X_test_fold = encoder_obj.transform(X_test_fold)
        #print("---------- after X_test_fold encoded ----------")
        #print(X_test_fold.head(10))

        y_pred = model_obj.predict(X_test_fold) 
        end_timer = time.time()
        elapsed_time = end_timer - start_timer
        f1_score_results.append(f1_score(y_pred, y_test_fold))
        elapsed_times.append(elapsed_time)
        #f1_score_mean.append(f1_score(y_pred, y_test_fold))
        # increment counter
        fold_counter+=1

    #print("------ f1 result ")
    #print(f1_score_results)
    f1_score_results_mean = mean(f1_score_results)
    avg_elapsed_time = mean(elapsed_times)
    
    #print(f1_score_results_mean)
    return f1_score_results_mean, avg_elapsed_time