#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas_datareader as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import statsmodels.api as smf
from datetime import datetime


# In[2]:


get_ipython().system('pip install pandas-datareader')


# In[4]:


df=pd.read_csv('exchange_rate.csv')


# In[5]:


df


# In[6]:


df.info()


# In[7]:


df['date']=pd.to_datetime(df['date'],format="%d-%m-%Y %H:%M")


# In[8]:


df.info()


# In[9]:


df.plot()


# In[10]:


df.set_index('date',inplace=True)


# In[11]:


df


# In[12]:


from statsmodels.tsa.stattools import adfuller


# In[13]:


df.plot()


# In[14]:


##check the stationarity of data


# In[15]:


#ACFplot AND PACFplot 


# In[16]:


def adf_test(series):
    result=adfuller(series)
    print('ADF Statistics: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    if result[1] <=0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis,timeseries has no unit root indicates Data is stationary")
    else:
        print("weak evidence against nuoll hypothesis,timeseries has a unit root indicates Data is non-stationary")


# In[17]:


adf_test(df['Ex_rate'])


# In[18]:


df['Ex_rate_first']=df['Ex_rate']-df['Ex_rate'].shift(1)


# In[19]:


df['Ex_rate_first'].isnull().sum()


# In[20]:


df['Ex_rate_first'].dropna()


# In[21]:


adf_test(df['Ex_rate_first'].dropna())


# In[22]:


df


# In[23]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[24]:


acf=plot_acf(df['Ex_rate_first'].dropna())


# In[25]:


pacf=plot_pacf(df['Ex_rate_first'].dropna())


# In[26]:


df['EX_rate_twelve']=df['Ex_rate']-df['Ex_rate'].shift(12)


# In[27]:


df['EX_rate_twelve']


# In[28]:


df


# In[29]:


acf=plot_acf(df['EX_rate_twelve'].dropna())


# In[30]:


pacf=plot_pacf(df['EX_rate_twelve'].dropna())


# In[31]:


from datetime import datetime,timedelta
train_dataset_end=datetime(2005,10,10)
test_dataset_end=datetime(2010,10,10)


# In[32]:


train_data=df[:train_dataset_end]
test_data=df[train_dataset_end+timedelta(days=1):test_dataset_end]


# In[33]:


##prediction
pred_start_date=test_data.index[0]
pred_end_date=test_data.index[-1]


# In[34]:


from statsmodels.tsa.arima_model import ARIMA


# In[35]:


train_data


# In[36]:


# save finalized model to file
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
import numpy


# In[37]:


model_ARIMA=ARIMA(train_data['Ex_rate'],order=(5,1,3))


# In[38]:


model_Arima_fit=model_ARIMA.fit()


# In[39]:


model_Arima_fit.summary()


# In[40]:


pred_start_date=test_data.index[0]
pred_end_date=test_data.index[-1]
print(pred_start_date)
print(pred_end_date)


# In[41]:


pred=model_Arima_fit.predict(start=pred_start_date,end=pred_end_date)
residuals=test_data['Ex_rate']-pred


# In[42]:


pred


# In[43]:


residuals


# In[44]:


model_Arima_fit.resid.plot(kind='kde')


# In[45]:


test_data['Predicted_ARIMA']=pred


# In[46]:


test_data[['Ex_rate','Predicted_ARIMA']].plot()


# In[47]:


##create a SARIMA model


# In[48]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[49]:


model_SARIMAX=SARIMAX(train_data['Ex_rate'],order=(3,0,6),seasonal_order=(0,1,0,12))


# In[50]:


model_SARIMAX_fit=model_SARIMAX.fit()


# In[51]:


model_SARIMAX_fit.summary()


# In[52]:


pred_Sarima=model_SARIMAX_fit.predict(start=datetime(2009,10,1),end=datetime(2010,10,10))
residuals=test_data['Ex_rate']-pred_Sarima


# In[53]:


model_SARIMAX_fit.resid.plot()


# In[54]:


test_data['Predicted_SARIMA']=pred


# In[55]:


test_data[['Ex_rate','Predicted_ARIMA','Predicted_SARIMA']].plot()


# In[56]:


def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# In[57]:


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing 


# In[58]:


ses_model = SimpleExpSmoothing(train_data['Ex_rate']).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = test_data.index[0],end = test_data.index[-1])
MAPE(pred_ses,test_data['Ex_rate']) 


# In[59]:


hwe_model_add_add = ExponentialSmoothing(train_data['Ex_rate'],seasonal="add",trend="add",seasonal_periods=12).fit() #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = test_data.index[0],end = test_data.index[-1])
MAPE(pred_hwe_add_add,test_data['Ex_rate']) 


# In[60]:


hwe_model_mul_add = ExponentialSmoothing(train_data['Ex_rate'],seasonal="mul",trend="add",seasonal_periods=12).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = test_data.index[0],end = test_data.index[-1])
MAPE(pred_hwe_mul_add,test_data['Ex_rate'])


# In[61]:


hwe_model_add_add = ExponentialSmoothing(df['Ex_rate'],seasonal="add",trend="add",seasonal_periods=12).fit()


# In[62]:


#Forecasting for next 10 time periods
hwe_model_add_add.forecast(10)


# In[64]:


from scipy import stats


# In[65]:


fig, ax=plt.subplots()
stats.probplot(residuals,dist="norm",plot=ax)
plt.show()


# In[ ]:




