#!/usr/bin/env python
# coding: utf-8

# Tittle: Modeling and forecasting the covid-19 pandemic in India.
# 
# Import libraries for the project

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
#!pip install plotly
#!pip install cufflinks
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
#import earthpy as et
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from matplotlib.pylab import rcParams
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot, iplot, init_notebook_mode
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
init_notebook_mode(connected=True)
cf.go_offline()


# Understanding the data

# In[2]:


#read the data File 1
covid19IndiaByState = pd.read_csv('C:/Users/LV/Desktop/ML_Project/covid_india_states.csv')
covid19IndiaByState['Total Confirmed cases']=covid19IndiaByState['Total Confirmed cases'].astype('int')
covid19IndiaByState['Cured']=covid19IndiaByState['Cured'].astype('int')
covid19IndiaByState['Death']=covid19IndiaByState['Death'].astype('int')
#keeping only required column (exclude first column)
covid19IndiaByState=covid19IndiaByState[['State','Total Confirmed cases','Cured','Death']]
covid19IndiaByState.head(5)


# In[3]:


#check for any missing value(s)
covid19IndiaByState.isnull().sum()


# In[4]:


#check data information
covid19IndiaByState.info()


# In[5]:


#check data type
covid19IndiaByState.tail(5)


# In[6]:


#read the data File 2
covid19IndiaCase = pd.read_csv('C:/Users/LV/Desktop/ML_Project/case_time_series.csv',parse_dates=['Date'],dayfirst=True)
covid19IndiaCase=covid19IndiaCase[['Date','Daily Confirmed','Total Confirmed','Daily Recovered','Total Recovered','Daily Deceased','Total Deceased']]
#infer_datetime_format=True
#CVO.drop('Unnamed: 0',axis = 1,inplace =True)
covid19IndiaCase.head(5)


# In[7]:


#check for any missing value(s)(IGNORE)
covid19IndiaCase.isnull().sum()


# In[8]:


#check data information(IGNORE)
covid19IndiaCase.info()


# Exploratory data analysis

# In[9]:


#Build line graph to understand covid-19 trend in India
fig = go.Figure()
fig.add_trace(go.Scatter(x=covid19IndiaCase['Date'], y=covid19IndiaCase['Total Confirmed'], name='Confirmed',
                         line=dict(color='yellow', width=4)))
fig.add_trace(go.Scatter(x=covid19IndiaCase['Date'], y=covid19IndiaCase['Total Recovered'], name='Recovered',
                         line=dict(color='green', width=4)))
fig.add_trace(go.Scatter(x=covid19IndiaCase['Date'], y=covid19IndiaCase['Total Deceased'], name='Deaths',
                         line=dict(color='red', width=4)))
fig.update_layout(plot_bgcolor = '#fff',
    title='Covid-19 Trend in India',
     yaxis=dict(
        title='Number of Cases Per Day')
    )
fig.update_xaxes(showline=True, linecolor='black', gridcolor='grey')
fig.update_yaxes(showline=True, linecolor='black', gridcolor='grey')

fig.show()


# In[10]:


#Current  cases (Pie chart)

rvd = covid19IndiaCase['Daily Recovered'].sum() #total recovered
dth = covid19IndiaCase['Daily Deceased'].sum()  #total death
cfm= covid19IndiaCase['Daily Confirmed'].sum()  #total confirmed
act=cfm-rvd-dth #total active = total confirmed - total recovered - total death

fig = go.Figure(data=[go.Pie(labels=['Active','Recovered','Death'],
                             values= [act,rvd,dth],hole =.3)])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=['grey', 'green','red'], line=dict(color='#FFFFFF', width=2)))
fig.update_layout(title_text='Covid-19 cases in India',plot_bgcolor='rgb(275, 270, 273)')
fig.show()


# In[11]:


temp = covid19IndiaCase.copy()
fig = go.Figure(data=[
go.Bar(name='Deceased', x=temp['Date'], y=temp['Daily Deceased'],marker_color='red'),
go.Bar(name='Recovered', x=temp['Date'], y=temp['Daily Recovered'],marker_color='green'),
go.Bar(name='Confirmed', x=temp['Date'], y=temp['Daily Confirmed'],marker_color='yellow')])
fig.update_layout(barmode='stack')
fig.update_traces(textposition='inside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.update_layout(title_text='COVID-19 Daily Cases, Recovery and Deaths in India',
                  plot_bgcolor='rgb(275, 270, 273)')
fig.update_xaxes(showline=True, linecolor='black')
fig.update_yaxes(showline=True, linecolor='black')
fig.show()


# In[12]:


#Confirmed in India (bar chart)
fig = px.bar(covid19IndiaCase, x="Date", y="Total Confirmed", barmode='group', height=600,color_discrete_sequence = ['yellow'])
fig.update_layout(title_text='Covid-19 Daily Total Confirmed Cases in India',plot_bgcolor='rgb(275, 270, 273)')
fig.update_xaxes(showline=True, linecolor='black')
fig.update_yaxes(showline=True, linecolor='black')
fig.show()


# In[13]:


#Death Cases in India (bar chart)
fig = px.bar(covid19IndiaCase, x="Date", y="Total Deceased", barmode='group', height=600,color_discrete_sequence = ['red'])
fig.update_layout(title_text='Covid-19 Daily Total Death Cases in India',plot_bgcolor='rgb(275, 270, 273)')
fig.update_xaxes(showline=True, linecolor='black')
fig.update_yaxes(showline=True, linecolor='black')
fig.show()


# In[14]:


#Recovered Cases in India (bar chart)
fig = px.bar(covid19IndiaCase, x="Date", y="Total Recovered", barmode='group', height=600,color_discrete_sequence = ['green'])
fig.update_layout(title_text='Covid-19 Daily Total Recovered Cases in India',plot_bgcolor='rgb(275, 270, 273)')
fig.update_xaxes(showline=True, linecolor='black')
fig.update_yaxes(showline=True, linecolor='black')
fig.show()


# In[15]:


#Total Confirmed cases by state
fig ,ax = plt.subplots(figsize= (12,8))
fig.set_facecolor("white")
indexedcovid19IndiaByState=covid19IndiaByState.set_index(['State'])

current = indexedcovid19IndiaByState.sort_values("Total Confirmed cases",ascending=False)
p = sns.barplot(ax=ax,x= current.index,y=current['Total Confirmed cases'])
p.set_xticklabels(labels = current.index,rotation=90)
p.set_xlabel("State",fontsize=16)
p.set_ylabel("Confirm cases",fontsize=16)

p.set_yticklabels(labels=(p.get_yticks()*1).astype(int))
plt.title("Total Confirmed cases by state in India",fontsize=20)


# In[16]:


#Confirmed,Recovered & Death figures for top 5 Total Recover cases
f, ax = plt.subplots(figsize=(12, 8))
data = covid19IndiaByState[['State','Total Confirmed cases','Cured','Death']]
data.sort_values('Cured',ascending=False,inplace=True)
data=data[0:5]
sns.set_color_codes("pastel")
sns.barplot(x="Total Confirmed cases", y="State", data=data,label="Total", color="yellow")
sns.barplot(x="Cured", y="State", data=data, label="Cured", color="green")
sns.barplot(x="Death", y="State", data=data, label="Death", color="red")
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 12000), ylabel="",xlabel="Cases")
ax.set_xlabel("Cases",fontsize=16)
ax.set_ylabel("States",fontsize=16)
sns.despine(left=True, bottom=True)
sns.set_style("whitegrid")
plt.title("Confirmed, Recovered & Death for top 5 state with total confirmed cases in India",fontsize=20)


# In[17]:


#Top 5 states with highest number of cured cases along with total confirmed & death (bar chart)
covid19IndiaByState.sort_values(by='Cured',ascending=False)[:5].iplot(kind='bar',x='State',
                                                                               color = ['yellow','green','red'],
                                                                               title='Top 5 States with total cured cases along with total confirm & death in India',
                                                                                xTitle='States',
                                                                               yTitle = 'Cases Count')


# In[18]:


#Top 5 states with highest number of confirm cases (bar chart)
top5State_DeathCases=covid19IndiaByState.sort_values(by='Total Confirmed cases',ascending=False)
top5State_DeathCases=top5State_DeathCases[0:5]
sns.set(rc={'figure.figsize':(15,10),"axes.labelsize":15})
sns.barplot(x="State",y='Total Confirmed cases',data=top5State_DeathCases,hue='State')
plt.title("Top 5 States with total confirmed cases in India",fontsize=20)
plt.show()


# In[19]:


#Top 5 states with highest number of death cases (bar chart)
top5State_DeathCases=covid19IndiaByState.sort_values(by='Death',ascending=False)
top5State_DeathCases=top5State_DeathCases[0:5]
sns.set(rc={'figure.figsize':(15,10),"axes.labelsize":15})
sns.barplot(x="State",y='Death',data=top5State_DeathCases,hue='State')
plt.title("Top 5 States with total death cases in India",fontsize=20)
plt.show()


# In[20]:


#Top 5 states with highest number of cured cases (bar chart)
top5State_CuredCases=covid19IndiaByState.sort_values(by='Cured',ascending=False)
top5State_CuredCases=top5State_CuredCases[0:5]
sns.set(rc={'figure.figsize':(15,10),"axes.labelsize":15})
sns.barplot(x="State",y='Cured',data=top5State_CuredCases,hue='State')
plt.title("Top 5 States with total cured cases in India",fontsize=20)
plt.show()


# From the bar plots,Maharashtra shows the highest number for total confirmed, death and cured cases.

# Objective 1: Correlation

# In[21]:


#correlation value
corr= covid19IndiaCase[['Total Confirmed','Total Recovered','Total Deceased']].corr()
print(corr)


# In[22]:


#correlation visualization
mask = np.triu(np.ones_like(corr,dtype = bool))
plt.figure(dpi=100)
plt.title('Correlation Analysis',fontsize=18)
sns.heatmap(corr,mask=mask,annot=True,lw=1,linecolor='white',cmap='Blues')
plt.xticks(rotation=0)
plt.yticks(rotation = 0)
plt.show()


# It can be shown that there is strong correlation between Total Confirmed, Total Recovered and Total Deceased cases.
# Use Granger Causality Tests to measure the presence of causality..

# In[23]:


#Granger Causality Tests
from statsmodels.tsa.stattools import grangercausalitytests


# In[24]:


grangercausalitytests(covid19IndiaCase[['Total Confirmed','Total Recovered']],maxlag=4);


# In[25]:


grangercausalitytests(covid19IndiaCase[['Total Confirmed','Total Deceased']],maxlag=4);


# Objective 2: Forecasting

# In[26]:


#show version of intalled packages
#pd.show_versions()


# In[27]:


#reducing the version to be able to use fbprophet 
#!pip install pystan==2.19.0.0


# In[28]:


#reducing the version to be able to use fbprophet
#!pip install pandas==1.0.4


# In[29]:


#reducing the version to be able to use fbprophet
#!pip install fbprophet==0.6.0


# In[30]:


#maintaining usable column only
IndiaForecast=covid19IndiaCase
IndiaForecast.drop(['Daily Confirmed','Daily Recovered','Daily Deceased'],axis = 1, inplace = True)
IndiaForecast.head(5)


# In[31]:


#check data information
IndiaForecast.info()


# In[32]:


#forecast Confirmed Cases
Confirmed = IndiaForecast.copy()
Confirmed = Confirmed[['Date','Total Confirmed']]
Confirmed.columns = ['ds','y']
print(Confirmed)


# In[33]:


#forecast Confirmed Cases for May & June 2020:print data up to which date
m = Prophet(interval_width=0.95)
m.fit(Confirmed)
futureconfirm = m.make_future_dataframe(periods=60)
futureconfirm .tail()


# In[34]:


#forecast Confirmed Cases for May & June 2020:print latest data value
forecastconfirm=m.predict(futureconfirm )
forecastconfirm[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[35]:


#forecast Confirm Cases for May & June 2020:plot the Death cases)
forecastconfirm=m.predict(forecastconfirm)
figC = plot_plotly(m, forecastconfirm)  # This returns a plotly Figure
figC.update_layout(autosize=False,
                  width= 750,
                  height= 800,
    title_text='<b>Covid-19 Confirm Forecast<b>',
    title_x=0.5,
    paper_bgcolor='white',
    plot_bgcolor = "yellow",)
figC.update_xaxes(showline=True, linecolor='black')
figC.update_yaxes(showline=True, linecolor='black')
figC.show()


# In[36]:


confirmed_component_plot =m.plot_components(forecastconfirm)


# In[37]:


#Confirmed cases Predicted vs Actual
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=IndiaForecast['Date'], y=IndiaForecast['Total Confirmed'],
                    mode='lines+markers',marker_color='Yellow',name='Actual'))
fig1.add_trace(go.Scatter(x=forecastconfirm['ds'], y=forecastconfirm['yhat_upper'],
                    mode='lines',marker_color='Grey',name='Predicted'))
fig1.update_layout(title_text = 'Confirmed cases Predicted vs Actual using prophet')

fig1.update_layout(plot_bgcolor='rgb(275, 270, 273)',width=600, height=600)
fig1.update_xaxes(showline=True, linecolor='black')
fig1.update_yaxes(showline=True, linecolor='black')


fig1.show()


# In[38]:


#calculate MSE
from fbprophet.diagnostics import cross_validation
Confirmed.cv=cross_validation(m, initial='50 days', period='13 days', horizon = '25 days')
Confirmed.cv.tail()


# In[39]:


#calculate MSE
from fbprophet.diagnostics import performance_metrics
Confirmed_performance = performance_metrics(Confirmed.cv)
Confirmed_performance.head(5)


# In[40]:


#plot
from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(Confirmed.cv, metric='mape')


# In[41]:


Confirmed_performance.mean()


# In[42]:


#forecast Death Cases
dth = IndiaForecast.copy()
Deaths = dth[['Date','Total Deceased']]
Deaths.columns = ['ds','y']
print(Deaths)


# In[43]:


#forecast Death Cases for May & June 2020:print data up to which date
m = Prophet(interval_width=0.95)
m.fit(Deaths)
futuredeath = m.make_future_dataframe(periods=60)
futuredeath.tail()


# In[44]:


#forecast Death Cases for May & June 2020:print latest data value
forecastdeath=m.predict(futuredeath)
forecastdeath[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[45]:


#forecast Death Cases for May & June 2020:plot the Death cases)
forecastdeath=m.predict(futuredeath)
figD = plot_plotly(m, forecastdeath)  # This returns a plotly Figure
figD.update_layout(autosize=False,
                  width= 750,
                  height= 800,
    title_text='<b>Covid-19 Death Forecast<b>',
    title_x=0.5,
    paper_bgcolor='white',
    plot_bgcolor = "mistyrose",)
figD.update_xaxes(showline=True, linecolor='black')
figD.update_yaxes(showline=True, linecolor='black')
figD.show()


# In[46]:


deaths_component_plot = m.plot_components(forecastdeath)


# In[47]:


#Death cases Predicted vs Actual
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=IndiaForecast['Date'], y=IndiaForecast['Total Deceased'],
                    mode='lines+markers',marker_color='red',name='Actual'))
fig2.add_trace(go.Scatter(x=forecastdeath['ds'], y=forecastdeath['yhat_upper'],
                    mode='lines',marker_color='Grey',name='Predicted'))
fig2.update_layout(title_text = 'Death cases Predicted vs Actual using prophet')
fig2.update_layout(plot_bgcolor='rgb(275, 270, 273)',width=600, height=600)
fig2.update_xaxes(showline=True, linecolor='black')
fig2.update_yaxes(showline=True, linecolor='black')
fig2.show()


# In[48]:


#calculate MSE
from fbprophet.diagnostics import cross_validation
Deaths.cv=cross_validation(m, initial='50 days', period='13 days', horizon = '25 days')
Deaths.cv.tail()


# In[49]:


#calculate MSE
from fbprophet.diagnostics import performance_metrics
Death_performance = performance_metrics(Deaths.cv)
Death_performance.head(5)


# In[50]:


#plot
from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(Deaths.cv, metric='mape')


# In[51]:


Death_performance.mean()

