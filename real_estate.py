#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pwd', '')


# In[2]:


import time
import random
from math import *
import operator
import pandas as pd
import numpy as np

# import plotting libraries
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set(font_scale=1.5)


# ### Import data

# In[3]:


df_train=pd.read_csv("C:/Users/USER/Documents/real_estate_dataSource/train.csv")


# In[4]:


df_test=pd.read_csv("C:/Users/USER/Documents/real_estate_dataSource/test.csv")


# In[5]:


df_train.columns


# In[6]:


df_test.columns


# In[7]:


len(df_train)


# In[8]:


len(df_test)


# In[9]:


df_train.head()


# In[10]:


df_test.head()


# In[11]:


df_train.describe()


# In[12]:


df_test.describe()


# In[13]:


df_train.info()


# In[14]:


df_test.info()


# ### 2 . Figure out the primary key and look for the requirement of indexing

# In[15]:


#UID is unique userID value in the train and test dataset. So an index can be created from the UID feature
df_train.set_index(keys=['UID'],inplace=True)#Set the DataFrame index using existing columns.
df_test.set_index(keys=['UID'],inplace=True)


# In[16]:


df_train.head(2)


# ### 3. Gauge the fill rate of the variables and devise plans for missing value treatment. Please explain explicitly the reason for the treatment chosen for each variable
# 

# In[17]:


#percantage of missing values in train set
missing_list_train=df_train.isnull().sum() *100/len(df_train)
missing_values_df_train=pd.DataFrame(missing_list_train,columns=['Percantage of missing values'])
missing_values_df_train.sort_values(by=['Percantage of missing values'],inplace=True,ascending=False)
missing_values_df_train[missing_values_df_train['Percantage of missing values'] >0][:10]
#BLOCKID can be dropped, since it is 100%missing values


# In[18]:


#percantage of missing values in test set
missing_list_test=df_test.isnull().sum() *100/len(df_train)
missing_values_df_test=pd.DataFrame(missing_list_test,columns=['Percantage of missing values'])
missing_values_df_test.sort_values(by=['Percantage of missing values'],inplace=True,ascending=False)
missing_values_df_test[missing_values_df_test['Percantage of missing values'] >0][:10]
#BLOCKID can be dropped, since it is 43%missing values


# In[19]:


df_train .drop(columns=['BLOCKID','SUMLEVEL'],inplace=True) #SUMLEVEL doest not have any predictive power and no variance


# In[20]:


df_test .drop(columns=['BLOCKID','SUMLEVEL'],inplace=True) #SUMLEVEL doest not have any predictive power


# In[21]:


# Imputing  missing values with mean
missing_train_cols=[]
for col in df_train.columns:
    if df_train[col].isna().sum() !=0:
         missing_train_cols.append(col)
print(missing_train_cols)


# In[22]:


# Imputing  missing values with mean
missing_test_cols=[]
for col in df_test.columns:
    if df_test[col].isna().sum() !=0:
         missing_test_cols.append(col)
print(missing_test_cols)


# In[23]:


# Missing cols are all numerical variables
for col in df_train.columns:
  if col in (missing_train_cols):
      df_train[col].replace(np.nan, df_train[col].mean(),inplace=True)


# In[24]:


# Missing cols are all numerical variables
for col in df_test.columns:
    if col in (missing_test_cols):
        df_test[col].replace(np.nan, df_test[col].mean(),inplace=True)


# In[25]:


df_train.isna().sum().sum()


# In[26]:


df_test.isna().sum().sum()


# ### Exploratory Data Analysis (EDA):
# #### Perform debt analysis. You may take the following steps:
# 

# a) Explore the top 2,500 locations where the percentage of households with a second mortgage is the highest and percent ownership is above 10 percent. Visualize using geo-map. You may keep the upper limit for the percent of households with a second mortgage to 50 percent.

# In[28]:


get_ipython().system('pip install pandasql')


# In[29]:


from pandasql import sqldf
q1 = "select place,pct_own,second_mortgage,lat,lng from df_train where pct_own >0.10 and second_mortgage <0.5 order by second_mortgage DESC LIMIT 2500;"
pysqldf = lambda q: sqldf(q, globals())
df_train_location_mort_pct=pysqldf(q1)


# In[30]:


df_train_location_mort_pct.head()


# In[31]:


import plotly.express as px
import plotly.graph_objects as go


# In[32]:


fig = go.Figure(data=go.Scattergeo(
    lat = df_train_location_mort_pct['lat'],
    lon = df_train_location_mort_pct['lng']),
    )
fig.update_layout(
    geo=dict(
        scope = 'north america',
        showland = True,
        landcolor = "rgb(212, 212, 212)",
        subunitcolor = "rgb(255, 255, 255)",
        countrycolor = "rgb(255, 255, 255)",
        showlakes = True,
        lakecolor = "rgb(255, 255, 255)",
        showsubunits = True,
        showcountries = True,
        resolution = 50,
        projection = dict(
            type = 'conic conformal',
            rotation_lon = -100
        ),
        lonaxis = dict(
            showgrid = True,
            gridwidth = 0.5,
            range= [ -140.0, -55.0 ],
            dtick = 5
        ),
        lataxis = dict (
            showgrid = True,
            gridwidth = 0.5,
            range= [ 20.0, 60.0 ],
            dtick = 5
        )
    ),
    title='Top 2,500 locations with second mortgage is the highest and percent ownership is above 10 percent')
fig.show()


# Use the following bad debt equation: Bad Debt = P (Second Mortgage ∩ Home Equity Loan) Bad Debt = second_mortgage + home_equity - home_equity_second_mortgage c) Create pie charts to show overall debt and bad debt

# In[33]:


df_train['bad_debt']=df_train['second_mortgage']+df_train['home_equity']-df_train['home_equity_second_mortgage']


# In[34]:


df_train['bins'] = pd.cut(df_train['bad_debt'],bins=[0,0.10,1], labels=["less than 50%","50-100%"])
df_train.groupby(['bins']).size().plot(kind='pie',subplots=True,startangle=90, autopct='%1.1f%%')
plt.axis('equal')

plt.show()
#df.plot.pie(subplots=True,figsize=(8, 3))


# Create Box and whisker plot and analyze the distribution for 2nd mortgage, home equity, good debt, and bad debt for different cities.

# In[35]:


cols=[]
df_train.columns


# In[36]:


#Taking Hamilton and Manhattan cities data
cols=['second_mortgage','home_equity','debt','bad_debt']
df_box_hamilton=df_train.loc[df_train['city'] == 'Hamilton']
df_box_manhattan=df_train.loc[df_train['city'] == 'Manhattan']
df_box_city=pd.concat([df_box_hamilton,df_box_manhattan])
df_box_city.head(4)


# In[37]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df_box_city,x='second_mortgage', y='city',width=0.5,palette="Set3")
plt.show()


# In[38]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df_box_city,x='home_equity', y='city',width=0.5,palette="Set3")
plt.show()


# In[39]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df_box_city,x='debt', y='city',width=0.5,palette="Set3")
plt.show()


# In[40]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df_box_city,x='debt', y='city',width=0.5,palette="Set3")
plt.show()


# ##### Manhattan has higher metrics compared to Hamilton

# #### Create a collated income distribution chart for family income, house hold income, and remaining income

# In[41]:


sns.distplot(df_train['hi_mean'])
plt.title('Household income distribution chart')
plt.show()


# In[42]:


sns.distplot(df_train['family_mean'])
plt.title('Family income distribution chart')
plt.show()


# In[43]:


sns.distplot(df_train['family_mean']-df_train['hi_mean'])
plt.title('Remaining income distribution chart')
plt.show()


# #### Income distribution almost has normality in its distrbution

# #### Perform EDA and come out with insights into population density and age. You may have to derive new fields (make sure to weight averages for accurate measurements):

# In[44]:


#plt.figure(figsize=(25,10))
fig,(ax1,ax2,ax3)=plt.subplots(3,1)
sns.distplot(df_train['pop'],ax=ax1)
sns.distplot(df_train['male_pop'],ax=ax2)
sns.distplot(df_train['female_pop'],ax=ax3)
plt.subplots_adjust(wspace=0.8,hspace=0.8)
plt.tight_layout()
plt.show()


# In[45]:


#plt.figure(figsize=(25,10))
fig,(ax1,ax2)=plt.subplots(2,1)
sns.distplot(df_train['male_age_mean'],ax=ax1)
sns.distplot(df_train['female_age_mean'],ax=ax2)
plt.subplots_adjust(wspace=0.8,hspace=0.8)
plt.tight_layout()
plt.show()


# #### a) Use pop and ALand variables to create a new field called population density

# In[46]:


df_train['pop_density']=df_train['pop']/df_train['ALand']


# In[47]:


df_test['pop_density']=df_test['pop']/df_test['ALand']


# In[48]:


sns.distplot(df_train['pop_density'])
plt.title('Population Density')
plt.show() # Very less density is noticed


# #### Use male_age_median, female_age_median, male_pop, and female_pop to create a new field called median age c) Visualize the findings using appropriate chart type

# In[49]:


df_train['age_median']=(df_train['male_age_median']+df_train['female_age_median'])/2
df_test['age_median']=(df_test['male_age_median']+df_test['female_age_median'])/2


# In[50]:


df_train[['male_age_median','female_age_median','male_pop','female_pop','age_median']].head()


# In[51]:


sns.distplot(df_train['age_median'])
plt.title('Median Age')
plt.show()
# Age of population is mostly between 20 and 60
# Majority are of age around 40
# Median age distribution has a gaussian distribution
# Some right skewness is noticed


# In[52]:


sns.boxplot(df_train['age_median'])
plt.title('Population Density')
plt.show() 


# #### Create bins for population into a new variable by selecting appropriate class interval so that the number of categories don’t exceed 5 for the ease of analysis.

# In[53]:


df_train['pop'].describe()


# In[54]:


df_train['pop_bins']=pd.cut(df_train['pop'],bins=5,labels=['very low','low','medium','high','very high'])


# In[55]:


df_train[['pop','pop_bins']]


# In[56]:


df_train['pop_bins'].value_counts()


# #### Analyze the married, separated, and divorced population for these population brackets

# In[57]:


df_train.groupby(by='pop_bins')[['married','separated','divorced']].count()


# In[58]:


df_train.groupby(by='pop_bins')[['married','separated','divorced']].agg(["mean", "median"])


# 1. Very high population group has more married people and less percantage of separated and divorced couples
# 2. In very low population groups, there are more divorced people

# #### Visualize using appropriate chart type

# In[59]:


plt.figure(figsize=(10,5))
pop_bin_married=df_train.groupby(by='pop_bins')[['married','separated','divorced']].agg(["mean"])
pop_bin_married.plot(figsize=(20,8))
plt.legend(loc='best')
plt.show()


# ##### Please detail your observations for rent as a percentage of income at an overall level, and for different states.

# In[60]:


rent_state_mean=df_train.groupby(by='state')['rent_mean'].agg(["mean"])
rent_state_mean.head()


# In[61]:


income_state_mean=df_train.groupby(by='state')['family_mean'].agg(["mean"])
income_state_mean.head()


# In[62]:


rent_perc_of_income=rent_state_mean['mean']/income_state_mean['mean']
rent_perc_of_income.head(10)


# In[63]:


#overall level rent as a percentage of income
sum(df_train['rent_mean'])/sum(df_train['family_mean'])


# #### Perform correlation analysis for all the relevant variables by creating a heatmap. Describe your findings.

# In[64]:


df_train.columns


# In[65]:


cor=df_train[['COUNTYID','STATEID','zip_code','type','pop', 'family_mean',
         'second_mortgage', 'home_equity', 'debt','hs_degree',
           'age_median','pct_own', 'married','separated', 'divorced']].corr()


# In[66]:


plt.figure(figsize=(20,10))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()


# 1.High positive correaltion is noticed between pop, male_pop and female_pop
# 2.High positive correaltion is noticed between rent_mean,hi_mean, family_mean,hc_mean

# 1. The economic multivariate data has a significant number of measured variables. The goal is to find where the measured variables depend on a number of smaller unobserved common factors or latent variables. 2. Each variable is assumed to be dependent upon a linear combination of the common factors, and the coefficients are known as loadings. Each measured variable also includes a component due to independent random variability, known as “specific variance” because it is specific to one variable. Obtain the common factors and then plot the loadings. Use factor analysis to find latent variables in our dataset and gain insight into the linear relationships in the data. Following are the list of latent variables:

# • Highschool graduation rates • Median population age • Second mortgage statistics • Percent own • Bad debt expense

# In[68]:


pip install factor_analyzer


# In[69]:


from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer


# In[70]:


fa=FactorAnalyzer(n_factors=5)
fa.fit_transform(df_train.select_dtypes(exclude= ('object','category')))
fa.loadings_


# Data Modeling : Linear Regression

# Build a linear Regression model to predict the total monthly expenditure for home mortgages loan. Please refer ‘deplotment_RE.xlsx’. Column hc_mortgage_mean is predicted variable. This is the mean monthly mortgage and owner costs of specified geographical location. Note: Exclude loans from prediction model which have NaN (Not a Number) values for hc_mortgage_mean.

# In[71]:


df_train.columns


# In[72]:


df_train['type'].unique()
type_dict={'type':{'City':1, 
                   'Urban':2, 
                   'Town':3, 
                   'CDP':4, 
                   'Village':5, 
                   'Borough':6}
          }
df_train.replace(type_dict,inplace=True)


# In[73]:


df_train['type'].unique()


# In[74]:


df_test.replace(type_dict,inplace=True)


# In[75]:


df_test['type'].unique()


# In[76]:


feature_cols=['COUNTYID','STATEID','zip_code','type','pop', 'family_mean',
         'second_mortgage', 'home_equity', 'debt','hs_degree',
           'age_median','pct_own', 'married','separated', 'divorced']


# In[77]:


x_train=df_train[feature_cols]
y_train=df_train['hc_mortgage_mean']


# In[78]:


x_test=df_test[feature_cols]
y_test=df_test['hc_mortgage_mean']


# In[79]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,accuracy_score


# In[80]:


x_train.head()


# In[81]:


sc=StandardScaler()
x_train_scaled=sc.fit_transform(x_train)
x_test_scaled=sc.fit_transform(x_test)


# Run a model at a Nation level. If the accuracy levels and R square are not satisfactory proceed to below step.

# In[82]:


linereg=LinearRegression()
linereg.fit(x_train_scaled,y_train)


# In[83]:


y_pred=linereg.predict(x_test_scaled)


# In[84]:


print("Overall R2 score of linear regression model", r2_score(y_test,y_pred))
print("Overall RMSE of linear regression model", np.sqrt(mean_squared_error(y_test,y_pred)))


# The Accuracy and R2 score are good, but still will investigate the model performance at state level

# Run another model at State level. There are 52 states in USA.

# Run another model at State level. There are 52 states in USA.

# In[86]:


for i in [20,1,45]:
    print("State ID-",i)
    
    x_train_nation=df_train[df_train['COUNTYID']==i][feature_cols]
    y_train_nation=df_train[df_train['COUNTYID']==i]['hc_mortgage_mean']
    
    x_test_nation=df_test[df_test['COUNTYID']==i][feature_cols]
    y_test_nation=df_test[df_test['COUNTYID']==i]['hc_mortgage_mean']
    
    x_train_scaled_nation=sc.fit_transform(x_train_nation)
    x_test_scaled_nation=sc.fit_transform(x_test_nation)
    
    linereg.fit(x_train_scaled_nation,y_train_nation)
    y_pred_nation=linereg.predict(x_test_scaled_nation)
    
    print("Overall R2 score of linear regression model for state,",i,":-" ,r2_score(y_test_nation,y_pred_nation))
    print("Overall RMSE of linear regression model for state,",i,":-" ,np.sqrt(mean_squared_error(y_test_nation,y_pred_nation)))
    print("\n")


# In[87]:


# To check the residuals


# In[88]:


residuals=y_test-y_pred
residuals


# In[89]:


plt.hist(residuals) # Normal distribution of residuals


# In[90]:


sns.distplot(residuals)


# In[91]:


plt.scatter(residuals,y_pred) # Same variance and residuals does not have correlation with predictor
# Independance of residuals

