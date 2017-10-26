
# coding: utf-8

# In[1]:


import numpy as np
from scipy import stats
import statsmodels.api as sm
from fbprophet import Prophet

def make_prophet_prediction(ts_train, prediction_period, withTransform=False):
    if withTransform == True:
        ts_train["y"] = np.log(ts_train["y"])
        
    model_prophet = Prophet()
    model_prophet.fit(ts_train)
    ts_predict = model_prophet.predict(model_prophet.make_future_dataframe(periods=prediction_period))
    
    if withTransform == True:
        ts_predict["yhat"] = np.exp(ts_predict["yhat"])
        ts_predict["yhat_upper"] = np.exp(ts_predict["yhat_upper"])
        ts_predict["yhat_lower"] = np.exp(ts_predict["yhat_lower"])
  
    ts_predict.loc[ts_predict["yhat"] < 0, "yhat"] = 0
    ts_predict.loc[ts_predict["yhat_upper"] < 0, "yhat"] = 0
    ts_predict.loc[ts_predict["yhat_lower"] < 0, "yhat"] = 0
    
    return (model_prophet, ts_predict)


def invboxcox(y, lmbda):
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda * y + 1) / lmbda))
    
def make_prophet_prediction_boxcox(ts_train, prediction_period):
    #train_df2 = train_df2.set_index('ds')
    ts_train['y'], lmbda_prophet = stats.boxcox(ts_train['y'])        
    
    model_prophet = Prophet()
    model_prophet.fit(ts_train)
    ts_predict = model_prophet.predict(model_prophet.make_future_dataframe(periods=prediction_period))
    
    ts_predict["yhat"] = invboxcox(ts_predict["yhat"], lmbda_prophet)
    ts_predict["yhat_upper"] = invboxcox(ts_predict["yhat_upper"], lmbda_prophet)
    ts_predict["yhat_lower"] = invboxcox(ts_predict["yhat_lower"], lmbda_prophet)
  
    ts_predict.loc[ts_predict["yhat"] < 0, "yhat"] = 0
    ts_predict.loc[ts_predict["yhat_upper"] < 0, "yhat"] = 0
    ts_predict.loc[ts_predict["yhat_lower"] < 0, "yhat"] = 0
    
    return (model_prophet, ts_predict)

def get_main_metrics(ts_test, ts_predict, start_date, end_date,  col_test, col_predict):
    ts_mae = abs(ts_test[start_date:end_date][col_test] - ts_predict[start_date:end_date][col_predict])
    ts_mape = abs(ts_mae / ts_test[start_date:end_date][col_test])

    print("MAE = {}".format(ts_mae.mean()))

    print("MAPE = {}".format(ts_mape.mean()))
    
    print("Sum of data for the testing period: %s " % str(ts_test[start_date:end_date][col_test].sum()))
    print("Sum of predicted data for the testing period: %s " % str(ts_predict[start_date:end_date][col_predict].sum()))


# In[2]:


import pandas as pd

def parser(x):
    return pd.datetime.strptime(x[:10], '%Y-%m-%d')

ts = pd.read_csv('d:\\Google Drive\\_My Work\\ml_projects\\ts_data\\ts_revenue.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)


# In[3]:


ts_train = ts[:"2017-08-01"]
ts_train = ts_train.reset_index()
ts_train.columns = ["ds", "y"]

(prophet_model, ts_predict) = make_prophet_prediction(ts_train, 30, withTransform=False)

ts_result = ts.to_frame().join(ts_predict.set_index("ds"), how="inner")

get_main_metrics(ts_result, ts_result, "2017-08-01", "2017-09-01", "col_value", "yhat")

prophet_model.plot(ts_predict)


# In[4]:


ts_train = ts[:"2017-08-01"]
ts_train = ts_train.reset_index()
ts_train.columns = ["ds", "y"]

(prophet_model, ts_predict) = make_prophet_prediction(ts_train, 30, withTransform=True)

ts_result = ts.to_frame().join(ts_predict.set_index("ds"), how="inner")

get_main_metrics(ts_result, ts_result, "2017-08-01", "2017-09-01", "col_value", "yhat")

prophet_model.plot(ts_predict)


# In[5]:


ts_train = ts[:"2017-09-01"]
ts_train = ts_train.reset_index()
ts_train.columns = ["ds", "y"]

(prophet_model, ts_predict) = make_prophet_prediction(ts_train, 30, withTransform=True)

ts_result = ts.to_frame().join(ts_predict.set_index("ds"), how="inner")

get_main_metrics(ts_result, ts_result, "2017-09-01", "2017-10-01", "col_value", "yhat")

prophet_model.plot(ts_predict)


# In[6]:


prophet_model.plot_components(ts_predict)


# In[7]:


import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

ts.plot()


# In[8]:


ts_viz = ts_predict.set_index("ds")
ts_viz["yhat_upper"].plot()
ts_viz["yhat_lower"].plot()
ts_viz["yhat"].plot()
ts.plot()


# In[9]:


ts_viz = ts_predict.set_index("ds")
#ts_viz["2017-08-01":"2017-10-01"]["yhat_upper"].plot()
#ts_viz["2017-08-01":"2017-10-01"]["yhat_lower"].plot()
ts_viz["2017-08-01":"2017-10-01"]["yhat"].plot()
ts["2017-08-01":"2017-10-01"].plot()


# In[10]:


ts_train = ts[:"2017-08-01"]
ts_train = ts_train.reset_index()
ts_train.columns = ["ds", "y"]

(prophet_model, ts_predict) = make_prophet_prediction_boxcox(ts_train, 30)

ts_result = ts.to_frame().join(ts_predict.set_index("ds"), how="inner")

get_main_metrics(ts_result, ts_result, "2017-08-01", "2017-09-01", "col_value", "yhat")

prophet_model.plot(ts_predict)


# In[11]:


ts_train = ts[:"2017-09-01"]
ts_train = ts_train.reset_index()
ts_train.columns = ["ds", "y"]

(prophet_model, ts_predict) = make_prophet_prediction_boxcox(ts_train, 30)

ts_result = ts.to_frame().join(ts_predict.set_index("ds"), how="inner")

get_main_metrics(ts_result, ts_result, "2017-09-01", "2017-10-01", "col_value", "yhat")

prophet_model.plot(ts_predict)


# In[12]:


ts_viz = ts_predict.set_index("ds")
#ts_viz["2017-08-01":"2017-10-01"]["yhat_upper"].plot()
#ts_viz["2017-08-01":"2017-10-01"]["yhat_lower"].plot()
ts_viz["2017-08-01":"2017-10-01"]["yhat"].plot()
ts["2017-08-01":"2017-10-01"].plot()


# In[ ]:




