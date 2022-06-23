import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
from pandas.core.indexes.api import get_objs_combined_axis
# import pmdarima as pm
# from pmdarima import model_selection
#import seaborn as sns 
from datetime import datetime
import typing
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
from statsmodels.tsa.api import ExponentialSmoothing, Holt

#from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
# upgrade to the the latest scikit-learn pip install -U scikit-learn
import random
import itertools

import sys

############################################## 
# will change existing code base that uses fc_propet to exponential Smoothing
# change date and value type requirements 
# change value to target & 
# target to ts_id
##############################################


# Dictionary for selecting period required by date_range function
# https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
periods = {365:'D',7:'D',12:'M',52:'W',4:'Q'}

class expo():
    """
    target field is the string name of the field, in the data frame you intend to forecast
    ts_id - requires that data series be pre serialized with a time series id (ts_id)
    field list - allows the procedure to match each ts_id with the field list, a list of fields representing
      dimensions for each single series. For example if you series are each store, each product, and each city, 
      then the field list is ['store', 'product','city']
    seasonal periods - the representation for frequency - daily, weekly, etc. stats models for expo smooth uses
      12, for yearly and 52 for weekly  
    """
    def __init__(self, 
                   df: pd.core.frame.DataFrame,
                   horizon: int,
                   date_variable: str, #typing.Union[int, str],
                   target_variable: str, #typing.Union[int, str],
                   field_list: list,
                   ts_id: str,
                   section_list: list = None,  # keep it for now
                   exog: np.array = None, # if supplied, must be of lenght of df.shape[0] or df.shape[0] + horizon
                   seasonal_periods: int = None,
                   **kwargs):
        self.date_variable = date_variable
        self.horizon = horizon
        self.exog = exog
        self.ts_id = ts_id
        self.df = df
        self.target_field = target_variable
        self.list_field = field_list.copy()
        self.list_field.append(self.ts_id)
        self.list_df = df[self.list_field].drop_duplicates().reset_index().copy()
        self.period = seasonal_periods

        # include serial numbers of combinations that don't have at least 24 observations
        self.short_list_keys = list()
        self.short_list_fields = dict()
        # the complement list - those combinations that were fitted
        self.mod_list_keys = list()
        self.models = dict()

        #  Ensure date variable is correct datetime format
        self.df[self.date_variable] = pd.to_datetime(self.df[self.date_variable])     
               
        # version of the data frame used through out methods and class
        self.df_ready = self.df.copy() # update from self.df_ready = df.copy()

       # Check if section list, passed in otherwise use full unique set
        self.section_list = section_list
        if self.section_list is None:
            self.section_list = df[ts_id].unique()


        # Create future date index - only need this calculated once
        self.fc_index = pd.date_range(start=self.df_ready[self.date_variable].max(), periods=self.horizon+1, freq=periods[self.period])[1:]
        
        #Exogenous variables handling   

        if self.exog is not None:
            print('Exogenous data included')
            self.exog_dict = dict() # declare exog dictionary

            if (self.exog.shape[-2] != self.df_ready[self.date_variable].unique().shape[0] + self.horizon) | (self.exog.shape[-1] != self.df_ready[self.ts_id].unique().shape[0]) :
                print("You Imbecile!!... Exogenous array dimension must match input data plus forecast horizon! exiting. Shameful! (:")
                #self.exog = None
                sys.exit()
                # fill in exog dictionary - tying each section to corresponding exogenous variable set
            for i, section in enumerate(self.df_ready.srl_num.unique()):
                # checks array dimmensions to see if single variable or multiple variables
                # are being used. If single exog shape is of form (b,c). If multiple, then
                # it's of form (a,b,c), where a is the number of exog variables, b is total 
                # observations [historical plus forecast horizon], and c is total # of series
                # single exog
                if len(self.exog.shape) == 2:
                    exog = self.exog[:,i]
                    exog = np.reshape(exog,(exog.shape[0],1))
                #multiple exog variables
                else:  
                    exog = self.exog[:,:,i].T    

                self.exog_dict[section] = exog


        forecast = pd.DataFrame()
        ## check if i / enumerate is needed
        for i, section in enumerate(self.section_list):
            temp = self.df_ready[self.df_ready[self.ts_id] == section].copy()
            print("Forecasting the following",temp[self.list_field].iloc[0,:],"...")
            if temp.shape[0] >= 2 * self.period: 
                # get single time series, strip to time series object

                # not needed if prep.py fill_blanks function is run. Can remove this 
                # in production
                temp.sort_values(by=self.date_variable,inplace=True)

                temp = temp[[self.date_variable,self.target_field]]
#                temp.columns = ['ds','y']
                fcast, m = self.get_train(temp,section)



                # if fcast.yhat[temp.shape[0]+1:].min() <0:
                #     fcast, m = self.train_logit(temp)        
                
                fcast[self.ts_id] = section
                forecast = forecast.append(fcast,ignore_index=True)
                self.mod_list_keys.append(section)
                # review the line of code below to see if this increases memory demands
                self.models[section] = m
            else:  ## not currently in use
                self.short_list_keys.append(section)
                self.short_list_fields[section] = temp[self.list_field].iloc[0,:]

        # include actuals into the forecast output data frame
        temp = self.df_ready[[self.ts_id,self.date_variable,self.target_field ]].copy()
        temp.columns = [self.ts_id,'ds','yact']
        forecast = forecast.merge(temp,how='left',on=['ds',self.ts_id])
        # add dimension columns back to forecast df, so see dimensions (columb fields you iterate on)
        temp_list = self.list_df.copy()
        forecast = forecast.merge(temp_list,how='inner', on=self.ts_id)    
        self.forecast = forecast
  
    def get_train(self,df,section):
        temp = df.copy()
        m = Prophet(changepoint_prior_scale = self.cp,seasonality_prior_scale=self.sp)
        # check for exogenous variable
        if self.exog is not None:
        # for multiple variables need to add for loop
            exog = self.exog_dict[section]
            exog_train = exog[:temp.shape[0],:]  

        # for each variable, generate a new exog variable column and pass in appropriate variable name
            for i in range(0,exog.shape[1]):
                m.add_regressor('exog'+str(i))
                temp['exog'+str(i)] = exog_train[:,i]

            # m.add_regressor('exog')
            # temp['exog'] = exog

            m.fit(temp)
            future = m.make_future_dataframe(periods=self.horizon,freq=self.freq)
            
            # for each variable, generate a new exog variable column and pass in appropriate variable name
            for i in range(0,exog.shape[1]):
                future['exog'+str(i)] = exog[:,i]

                #future['exog'] = self.exog 

        else: # no exog ... the default state
            m.fit(temp)
            future = m.make_future_dataframe(periods=self.horizon,freq=self.freq)
    
        fcast = m.predict(future) 

        # set floor at 0 for negative values
        fcast.yhat = np.where(fcast.yhat<0,0,fcast.yhat)
        return fcast,m

    def plot(self,srl_num: int = None):
        fig, ax = plt.subplots(figsize=(20,10))
        section = srl_num
        # prepare data objects that go into plot commands
        temp = self.forecast[self.forecast.srl_num == section].copy()
        temp_act = temp[temp.ds <= self.df_ready[self.date_variable].max()].yact.to_numpy()
        temp_fc = temp[temp.ds > self.df_ready[self.date_variable].max()][['yhat_lower','yhat_upper','yhat']]
        t_shape =temp_fc.shape
        temp_fc = temp_fc.to_numpy().reshape(t_shape)
        temp_fc = np.append(np.repeat(temp_act,3).reshape((temp_act.shape[0],3))[-1,:].reshape((1,3)),temp_fc,axis=0)
        ds_act = temp[temp.ds <= self.df_ready[self.date_variable].max()]['ds']
        ds_fc = temp[temp.ds >= self.df_ready[self.date_variable].max()]['ds']

        ax.plot(ds_act,temp_act,label="Actuals")
        ax.plot(ds_fc,temp_fc[:,2],'--',label="Forecast")

        ax.fill_between(ds_fc, temp_fc[:,0], temp_fc[:,1],alpha=0.1, color='b')

        plt.xlabel('Date')
        plt.ylabel(self.target_field)
        plt.legend()

        tmp = temp[self.list_field].drop_duplicates()
        # create plot title
        str1 = "" 
        for i, col in enumerate(tmp.iloc[0,:]):
            if i == len(tmp.columns) -1:
                str1 += '\''+col+'\''
            else:
                str1 += '\''+col+'\','
        str1
        plt.title(str1)

    def get_cv_df(self,start:int,step:int,srl_num: int = None,horizon:int=None,cp: float = .05,sp: float =10):        
        """if srl_num passed in, filter and get cv_df for just that selection
           otherwise loop though all srl_num options.
        """
        print('cp is',cp)
        print('sp is',sp)
        cv_df = pd.DataFrame()
        if srl_num is not None:
            #srl_num = srl_num
            temp = self.df_ready[self.df_ready[self.ts_id] == srl_num].copy()
            temp = self.create_cv_df(temp,start,step,horizon,cp,sp)
            temp[self.ts_id] = srl_num
            cv_df = cv_df.append(temp,ignore_index=True)
        else:
            for section in self.df_ready[self.ts_id].unique():
                temp = self.df_ready[self.df_ready[self.ts_id] == section].copy()
                temp = self.create_cv_df(temp,section,start,step,horizon,cp,sp)
                temp[self.ts_id] = section
                cv_df = cv_df.append(temp,ignore_index=True)
        # added this line to add f_list to output cv_df
        cv_df = cv_df.merge(self.list_df,how='left',on = self.ts_id)
        return cv_df

    def create_cv_df(self,df,section,start:int,step:int,horizon:int=None,cp : float= .05,sp : float=10):
        """function creates each single combination of cv_df
        """
        temp = df.copy()
        # debug 2
        print('cp is',cp)
        print('sp is',sp)
        cv_df = pd.DataFrame()
        end = temp.shape[0]

        if horizon is None:
            horizon = self.horizon

        while start < end:
            temp = df.iloc[:start,:]
            temp = temp[[self.date_variable,self.target_field]]
            temp.columns = ['ds','y']
            print('cutoff is:',start)
            temp_horizon = min(horizon,end-start)
            temp_end = start+temp_horizon
            print('forecasting to:',temp_end)

            fcast = self.train_cv(temp,section,start,temp_end,temp_horizon,cp,sp)

            # old code to check for negative predicitons
            # if fcast.yhat[-step:].min() < 0:
            #     fcast = self.train_cv_logit(temp,section,start,temp_end,temp_horizon,cp,sp)

            cv_df_temp = fcast[['ds','yhat','yhat_lower','yhat_upper']].tail(temp_horizon)
            cv_df_temp['y'] = df[self.target_field][start:start+temp_horizon].to_numpy()
            cv_df_temp['cutoff'] = temp.ds.max()
            cv_df = cv_df.append(cv_df_temp,ignore_index=True)
            start += step
        return cv_df

    def train_cv(self,df,section,start:int,temp_end:int,temp_horizon:int,cp:float,sp:float):
        """
        This function is specifically for training for cross validation
        """
        temp = df.copy()
        m = Prophet(changepoint_prior_scale = cp,seasonality_prior_scale=sp)

        print('start',start)

        if self.exog is not None:
            exog = self.exog_dict[section]
            exog_train = exog[:start,:]  
            exog_val = exog[:start+temp_horizon,:]

            # add columns and exog values to training set
            for i in range(0,exog.shape[1]):
                m.add_regressor('exog'+str(i))
                temp['exog'+str(i)] = exog_train[:,i]

            m.fit(temp)
            future = m.make_future_dataframe(periods=temp_horizon,freq=self.freq)
            # future['exog'] = exog[:start+temp_horizon]

            # add columns and exog values to forecast dataframe            
            for i in range(0,exog.shape[1]):
                future['exog'+str(i)] = exog_val[:,i]

        else:
            m.fit(temp)
            future = m.make_future_dataframe(periods=temp_horizon,freq=self.freq)
        fcast = m.predict(future)

        # set floor at 0 for negative values
        fcast.yhat = np.where(fcast.yhat<0,0,fcast.yhat)

        return fcast

    def get_metrics_summary(self,df,section: str = None, pivot_field: str=None):
        if pivot_field is None:
            pivot_field = self.ts_id

        df = df.groupby([df.columns[0],pivot_field]).sum().reset_index().copy()
        # debug
        # print(df.head(10))
        metrics = []
        
        for comb in df[pivot_field].unique():
            print('comb',comb)    
            temp = df[df[pivot_field] == comb].copy()
            metrics = self.get_metrics_detail(temp,metrics,pivot_field,comb)

        #aggr = df.drop(pivot_field,axis=1).groupby(['cutoff','ds']).sum().reset_index().copy()
        temp2 = df.groupby('ds').sum().reset_index()
        y_true = temp2.y
        y_pred = temp2.yhat
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true,y_pred)
        mae = mean_absolute_error(y_true,y_pred)
        metrics.append({pivot_field:'Total','rmse':rmse,'mape':mape,'mae':mae})

        metrics_df = pd.DataFrame(metrics)
        metrics_df = metrics_df.merge(self.list_df,how='left',on = self.ts_id)

        return metrics_df
        # if detail:
        #     return metrics_df
        # else:
        #     return metrics_df

    def get_metrics_detail(self,df,metrics,pivot_field,section):

        print('type',type(df))
        print(df.head())
        
        y_true = df.y
        y_pred = df.yhat

        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true,y_pred)
        mae = mean_absolute_error(y_true,y_pred)

        metrics.append({pivot_field:section,'rmse':rmse,'mape':mape,'mae':mae})

        return metrics    