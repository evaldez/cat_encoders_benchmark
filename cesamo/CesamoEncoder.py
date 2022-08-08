# set log
from nis import cat
from config.logging import set_logger
log=set_logger()
# CESAMO ORIGINAL
import random
import pandas as pd
from random import randint
import numpy
import math
import logging.config
# Shapiro-Wilk Test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import shapiro
import random
from random import randint
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import validation
import numpy as np
import os
import json
from scipy.stats import zscore
import scipy.stats as stats


class CesamoEncoder():
    def __init__(self, cols, run_with_cache=True):
        mapping = None
        self.encoded_metadata = {}
        self.cols = cols
        #self.min_error = 0.01269614407
        self.stop_on_accepted_error=True
        #self.min_error = 0.05
        self.min_error = 0.5
        self.iter_limit = 10000
        #self.iter_limit = 1000

        self.n_el_error_list=30
        self.least_el_in_avg_error_list=3
        self.code_digits=5
        self.code_and_errors={}
        self.run_with_cache=run_with_cache
        self.cache_base_path = './cesamo/cache/'
        self.min_el_in_each_decil = 5
        self.p_value = 0.05

    def fit(self, X, y=None, dataset_name=None, store_in_cache=True):
        """
            X data is expected to be scaled in values between 0 and 1
        """
        # check for cache
        dataset_name=None
        if "DATASET_NAME" in os.environ:
            dataset_name=os.environ["DATASET_NAME"]
        """
        cache = self.check_for_cache(os.environ["DATASET_NAME"])
        if cache:
            self.code_and_errors = cache
            return self
        """
        stop_condition = False
        all_features=X.columns
        log.debug(f'all_features: {all_features}')
        log.debug(f'catf: {self.cols}')
        for catf in self.cols:
            code_and_errors_tmp_list=[]
            error_list=[]
            error_avg_list=[]
            final_code_and_errors={}
            iteration=0
            is_gaussian=False
            log.debug(f'Applying CESAMO to cat att: {catf}')
            while not stop_condition:
                # make a copy of the trainset
                X_tmp = X.copy()
                # set a list of all attributes and remove current categorical attribute to convert
                features_tmp = all_features.tolist().copy()
                features_tmp.remove(catf)
                # Randomly select variable j (j!=i), (is the variable to be approximated)
                secondary_var = random.choice(features_tmp)
                #log.debug(f'selected secondary_var: {secondary_var}')
                # Assign random values to all instances of catf
                # if second var is cat, assign random codes
                if secondary_var in self.cols:
                    #log.debug(f"The second var is cat, assigning random codes")
                    # change every attribute for the col to random numbers
                    X_tmp, dead_var = self.convert_catcol_to_random(X_tmp, secondary_var)

                # Replace current cat att by proposed codes
                X_tmp, map_catin_code = self.convert_catcol_to_random(X_tmp, catf)

                #log.debug(f'map_catin_code: {map_catin_code}')
                # set dependent and independent var
                independent_var = X_tmp[[secondary_var]].copy()
                dependent_var = X_tmp[[catf]].copy()

                # Apply Polynomial feature expansion in order to apply an polynomial regression
                poly_features = PolynomialFeatures(degree=11, include_bias=False)
                X_poly = poly_features.fit_transform(independent_var)

                # remove even degrees in polinomial 
                X_poly = self.keep_odd_degrees(X_poly)
                lin_reg = LinearRegression()

                # approximate cat_at as a function of the randomly selected secondary variable
                lin_reg.fit(X_poly, dependent_var)
                
                # Measure RMSE 
                # predictions must be done on unseen data
                predictions = lin_reg.predict(X_poly)
                lin_mse = mean_squared_error(dependent_var, predictions)
                lin_rmse = np.sqrt(lin_mse)
                #log.debug(f'RMSE: {lin_rmse}')
                # store codes and errors
                code_and_errors_tmp_list.append({
                    "codes":map_catin_code,
                    "error":lin_rmse
                })
                if (lin_rmse < self.min_error) and self.stop_on_accepted_error:
                    log.info(f'acceptable error found {lin_rmse}')
                    final_code_and_errors = self.get_best_codes(code_and_errors_tmp_list)
                    log.info(f'final_code_and_errors: {final_code_and_errors}')

                    # break while loop
                    break
                
                # Store the errors to calculate if the distribution is Gaussian
                error_list.append(lin_rmse)
                # check if error list has at least N elements
                if len(error_list) == self.n_el_error_list :
                    error_avg_list.append(self.calc_avg(error_list))
                    # check if error avg list has at least N elements
                    if len(error_avg_list) >= self.least_el_in_avg_error_list:
                        #is_gaussian = self.is_gaussian_dis(error_avg_list)
                        #log.debug(f"Error_avg_list: { str(len(error_avg_list))}")
                        is_gaussian = self.check_if_gaussian(error_avg_list)

                        if is_gaussian:
                            log.info("Normal distriution of error codes has been reached")
                            final_code_and_errors = self.get_best_codes(code_and_errors_tmp_list)
                            log.info(f'final_code_and_errors: {final_code_and_errors}')
                            # break while loop
                            break

                    # empty list
                    error_list=[]


                if iteration % 1000 == 0: 
                    log.info(f"iteration: {iteration}")
                    
                iteration+=1

            # end of while
            self.code_and_errors[catf]=final_code_and_errors
        
        if store_in_cache and dataset_name: 
            self.save_in_cache(dataset_name, self.code_and_errors)

        return self

    def get_code(self, n):
        """
        function to generate a random decimal code of n digits after the point
        """
        return round(random.random(), n)

    def convert_catcol_to_random(self, df, col):
        # get every cat instance
        instances=df[col].unique()
        # for every instance, propose a code and set the map cat_instance - code
        map={}
        for inx, val in enumerate(instances):
            # simplified map
            map[val] = self.get_code(self.code_digits)
        # replace in df the map
        df[col] =  df[col].map(map)
        return df, map

    def calc_avg(self, numbers):
        """
        Calculate average
        """
        return numpy.mean(numbers) 

    # normality test
    def check_if_gaussian(self, data):
        """
            Determine if the distribution is gaussian
        """
        is_gaussian = False
        d1 = - 1.28 #(from 0 to 10.03 = 10.03)
        d2 = - 0.84 #(from 10.03 to 29.95= 10.02)
        d3 = - 0.52 #(from 29.95 to 40.05= 10.02)
        d4 = - 0.25 #(from 29.95 to 40.05= 10.02)
        d5 = 0      #()
        d6 = 0.25 #(from 0 to 10.03 = 10.03)
        d7 = 0.52 #(from 10.03 to 29.95= 10.02)
        d8 = 0.84 #(from 29.95 to 40.05= 10.02)
        d9 = 1.28 #(from 29.95 to 40.05= 10.02)
        # d10 = >d9      #()
        # iterate over data
        decils={
            "d1":[], "d2":[], "d3":[], "d4":[], "d5":[], 
            "d6":[], "d7":[], "d8":[], "d9":[], "d10":[]
        }
        zscore_data = zscore(data)
        for num in zscore_data:
            if num < d1:
                decils['d1'].append(num)
            elif d1 <= num < d2  :
                decils['d2'].append(num)
            elif d2 <= num < d3  :
                decils['d3'].append(num)    
            elif d3 <= num < d4  :
                decils['d4'].append(num)    
            elif d4 <= num < d5  :
                decils['d5'].append(num)   
            elif d5 <= num < d6  :
                decils['d6'].append(num)    
            elif d6 <= num < d7  :
                decils['d7'].append(num)  
            elif d7 <= num < d8  :
                decils['d8'].append(num)  
            elif d8 <= num < d9  :
                decils['d9'].append(num)  
            elif num >= d9 :
                decils['d10'].append(num) 
            else:
                raise Exception("Z-score didnt fit in decils: ", num)
        
        elements_per_decil = []
        for key, value in decils.items():
            elements_per_decil.append(len(value))
        log.debug(f"Elements_per_decil: {elements_per_decil}")
        for key, value in decils.items():
            # if there are less than min_el_in_each_decil, we cannot apply yet the chi squared test
            if len(value) < self.min_el_in_each_decil: 
                #log.debug(f'Some decil doesnt have yet the minimun number of elements: {self.min_el_in_each_decil}')
                # return is gaussian false
                return False

        # At this point, all elements have at least 5 elements
        # Apply chi square tests
        chi_square_test_result = stats.chisquare(f_obs=elements_per_decil)
        log.debug("Chisquare test value: {chi_square_test_result[1]}")
        if chi_square_test_result[1] > self.p_value:
            return True
        else: 
            return False

    def get_best_codes(self, codes_and_errors):
        """
            get best codes on based the lowest errors
        """
        smallest_error_val=None
        smallest_error_index=None
        for index, item in enumerate(codes_and_errors):

            if index==0: 
                smallest_error_val=item['error']
                smallest_error_index=index
                continue
            if smallest_error_val > item['error']: 
                smallest_error_val = item['error']
                smallest_error_index = index
        return codes_and_errors[smallest_error_index]

    def transform(self, X, y=None):
        """
            transform cat attributes to numerical values
        """ 
        X_tmp = X.copy()
        for cat_col in self.cols:
            log.debug(f'self.code_and_errors[cat_col]["codes"]:{self.code_and_errors[cat_col]["codes"]}')
            log.debug(f'cat_col {cat_col}')
            X_tmp[cat_col]= X_tmp[cat_col].map(str)
            X_tmp[cat_col] = X_tmp[cat_col].map( self.code_and_errors[cat_col]['codes'] )
        return X_tmp

    def check_for_cache(self, dataset_name):
        log.debug(f"checking if exist file for dataset_name: {dataset_name}" )
        # check in cache dir if dataset match the json file.
        cache_base_path = self.cache_base_path
        files = os.listdir(cache_base_path)
        files_name = [ x.replace('.json','') for x in files ]
        
        # if not, return empty dict
        if dataset_name not in files_name: return {}

        json_to_load = cache_base_path + dataset_name + '.json'
        # if exists, load the json into a dict
        with open(json_to_load, 'r') as jsonFile:
            jsonObject = json.load(jsonFile)
            jsonFile.close()
        
        log.info("codes and error founds")

        return jsonObject
    
    def save_in_cache(self, dataset_name, codes_and_error):
        log.info("Saving in cache")
        # save in a $$data_name$$.json the codes and erros
        file_name = self.cache_base_path + dataset_name + '.json'
        with open(file_name, 'w') as fp:
            json.dump(codes_and_error, fp)
        return

    def keep_odd_degrees(self, X):
        X_poly=X
        X_poly_odds=[]
        for idx,item in enumerate(X_poly):
            X_poly_odds.append([])
            for idx_2_loop,item_2_loop in enumerate(X_poly[idx]):
                if ((idx_2_loop+1) % 2) == 1: X_poly_odds[idx].append(item_2_loop)
        return X_poly_odds



    def set_logger():
        level='DEBUG'
        # env LOG_LEVEL var will overwrite the above value
        if "LOG_LEVEL" in os.environ:
            level=os.environ["LOG_LEVEL"]

        LOGGING_CONFIG = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default_formatter': {
                    'format': '[%(levelname)s:%(asctime)s] %(message)s'
                },
            },
            'handlers': {
                'stream_handler': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'default_formatter',
                },
            },
            'loggers': {
                'basic_logger': {
                    'handlers': ['stream_handler'],
                    'level': level,
                    'propagate': True
                }
            }
        }
        logging.config.dictConfig(LOGGING_CONFIG)
        logger = logging.getLogger('basic_logger')
        return logger