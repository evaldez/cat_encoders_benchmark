# set log
import logging.config
from config.logging import set_logger
log=set_logger()
import logging.config

import random
import pandas as pd
from random import randint
import numpy
import math
import os
import json



class RandomEncoder():
    def __init__(self, cols):
        mapping = None
        self.encoded_metadata = {}
        self.cols = cols
        self.code_digits=5
        self.map_codes_and_instances={}

    def fit(self, X, y=None):
        """
            Generate random values (0,1] for every categorical instance 
        """
        
        log.debug(f'catf: {self.cols}')
        for catf in self.cols:
            log.debug(f'generating codes for catf: {catf}')
            X_tmp = X.copy()
            # Replace current cat att by proposed codes
            X_tmp, map_catin_code = self.convert_catcol_to_random(X_tmp, catf)
            log.debug(f"map_catin_code: {map_catin_code}")
            final_codes_and_instances={}
            final_codes_and_instances['codes'] = map_catin_code
            self.map_codes_and_instances[catf] = final_codes_and_instances
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

    def transform(self, X, y=None):
        """
            transform cat attributes to numerical values
        """ 
        log.debug("transforming data")
        X_tmp = X.copy()
        for cat_col in self.cols:
            log.debug(f'self.map_codes_and_instances[cat_col]["codes"]:{self.map_codes_and_instances[cat_col]["codes"]}')
            log.debug(f'cat_col: {cat_col}')
            X_tmp[cat_col]= X_tmp[cat_col].map(str)
            X_tmp[cat_col] = X_tmp[cat_col].map( self.map_codes_and_instances[cat_col]['codes'] )
        return X_tmp


    def set_logger():
        level='DEBUG'
        # env LOG_LEVEL var will overwrite the above value
        #if "LOG_LEVEL" in os.environ:
        #    level=os.environ["LOG_LEVEL"]

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