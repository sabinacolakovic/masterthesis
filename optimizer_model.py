
import csv
import pandas as pd
## for online data
####################
import numpy as np
import math
from scipy.stats.mstats import gmean
import statistics
from scipy.stats import norm
import numpy as geek
import scipy.optimize as sco 
from scipy.optimize import minimize

import matplotlib.pyplot as plt

#plt.style.use('fivethirtyeight')
#np.random.seed(777)
#%%

'''Load Data set'''

 

with open('fonder_10_new.csv','r') as csv_file:
    csv_reader= csv.reader(csv_file)    
    #for line in csv_reader:
        #print(line)
       

#Store the data
new_cols = ['date', 'funds','prices']
df= pd.read_csv('fonder_10_new.csv',names=new_cols)
#%%

## df['date'] = pd.to_datetime.strptime(df['date'],'%y-%m-%d') #convert dates from / to -
df['date'] = pd.to_datetime(df['date']) #convert dates from / to -

#%%

## df['date'].astype(np.int64) #convert to integer
df['date'] = df['date'].dt.strftime('%Y%m%d').astype(int) #convert to integer
df.info() #info about the variable types

#%%

#set the funds as the index
#df.set_index('funds',inplace=False)
df['fund index']=None
n=1
i=1 #rows
#df['fund index']=df.loc[:,3]=n
#%%

#make fund names to integers names based on dates

for i in range(len(df['date'])-1):

    df.loc[i, 'fund index']=n

    if df['date'][i] > df['date'][i+1]:

        n=n+1
  
#%% 
#Make multiple small dataframe for each fund

result_df1 = df.groupby('fund index').get_group(1)
print(result_df1)

result_df2 = df.groupby('fund index').get_group(2)
print(result_df2)

result_df3 = df.groupby('fund index').get_group(3)
print(result_df3)

result_df4 = df.groupby('fund index').get_group(4)
print(result_df4)

result_df5 = df.groupby('fund index').get_group(5)
print(result_df5)

result_df6 = df.groupby('fund index').get_group(6)
print(result_df6)

result_df7 = df.groupby('fund index').get_group(7)
print(result_df7)

result_df8 = df.groupby('fund index').get_group(8)
print(result_df8)

result_df9 = df.groupby('fund index').get_group(9)
print(result_df9)

result_df10 = df.groupby('fund index').get_group(10)
print(result_df10)


#%%
# daily returns (log-returns)
rets1 = np.log((result_df1.loc[:,'prices']) / (result_df1.loc[:,'prices']).shift(1))
rets2 = np.log((result_df2.loc[:,'prices']) / (result_df2.loc[:,'prices']).shift(1))
rets3 = np.log((result_df3.loc[:,'prices']) / (result_df3.loc[:,'prices']).shift(1))
rets4 = np.log((result_df4.loc[:,'prices']) / (result_df4.loc[:,'prices']).shift(1))
rets5 = np.log((result_df5.loc[:,'prices']) / (result_df5.loc[:,'prices']).shift(1))
rets6 = np.log((result_df6.loc[:,'prices']) / (result_df6.loc[:,'prices']).shift(1))
rets7 = np.log((result_df7.loc[:,'prices']) / (result_df7.loc[:,'prices']).shift(1))
rets8 = np.log((result_df8.loc[:,'prices']) / (result_df8.loc[:,'prices']).shift(1))
rets9 = np.log((result_df9.loc[:,'prices']) / (result_df9.loc[:,'prices']).shift(1))
rets10 = np.log((result_df10.loc[:,'prices']) / (result_df10.loc[:,'prices']).shift(1))

#%%
#Daily returns
df['daily returns']=df['prices'].pct_change() #daily returnn column
#%%
#make a column with non negative and 0 values for the GM return
df['GM return']=df['daily returns']+1

#%%
#geometric mean return for each dataframe
result_df1['daily returns']=result_df1['prices'].pct_change() 
result_df1['GM return']=result_df1['daily returns']+1
result_df2['daily returns']=result_df2['prices'].pct_change() 
result_df2['GM return']=result_df2['daily returns']+1
result_df3['daily returns']=result_df3['prices'].pct_change() 
result_df3['GM return']=result_df3['daily returns']+1
result_df4['daily returns']=result_df4['prices'].pct_change() 
result_df4['GM return']=result_df4['daily returns']+1
result_df5['daily returns']=result_df5['prices'].pct_change() 
result_df5['GM return']=result_df5['daily returns']+1
result_df6['daily returns']=result_df6['prices'].pct_change() 
result_df6['GM return']=result_df6['daily returns']+1
result_df7['daily returns']=result_df7['prices'].pct_change() 
result_df7['GM return']=result_df7['daily returns']+1
result_df8['daily returns']=result_df8['prices'].pct_change() 
result_df8['GM return']=result_df8['daily returns']+1
result_df9['daily returns']=result_df9['prices'].pct_change() 
result_df9['GM return']=result_df9['daily returns']+1
result_df10['daily returns']=result_df10['prices'].pct_change() 
result_df10['GM return']=result_df10['daily returns']+1

#%%
#Geometric Mean rate of return
geometricMean1=gmean(result_df1.loc[:,'GM return'].dropna())-1
geometricMean2=gmean(result_df2.loc[:,'GM return'].dropna())-1
geometricMean3=gmean(result_df3.loc[:,'GM return'].dropna())-1
geometricMean4=gmean(result_df4.loc[:,'GM return'].dropna())-1
geometricMean5=gmean(result_df5.loc[:,'GM return'].dropna())-1
geometricMean6=gmean(result_df6.loc[:,'GM return'].dropna())-1
geometricMean7=gmean(result_df7.loc[:,'GM return'].dropna())-1
geometricMean8=gmean(result_df8.loc[:,'GM return'].dropna())-1
geometricMean9=gmean(result_df9.loc[:,'GM return'].dropna())-1
geometricMean10=gmean(result_df10.loc[:,'GM return'].dropna())-1

#%%
#Arithmetic Mean return
arithmeticMean1=statistics.mean(result_df1.loc[:,'daily returns'].dropna())
arithmeticMean2=statistics.mean(result_df2.loc[:,'daily returns'].dropna())
arithmeticMean3=statistics.mean(result_df3.loc[:,'daily returns'].dropna())
arithmeticMean4=statistics.mean(result_df4.loc[:,'daily returns'].dropna())
arithmeticMean5=statistics.mean(result_df5.loc[:,'daily returns'].dropna())
arithmeticMean6=statistics.mean(result_df6.loc[:,'daily returns'].dropna())
arithmeticMean7=statistics.mean(result_df7.loc[:,'daily returns'].dropna())
arithmeticMean8=statistics.mean(result_df8.loc[:,'daily returns'].dropna())
arithmeticMean9=statistics.mean(result_df9.loc[:,'daily returns'].dropna())
arithmeticMean10=statistics.mean(result_df10.loc[:,'daily returns'].dropna())
#%%
#daily and annual variance , assuming 253 business days
var1_daily =result_df1['daily returns'].var()
var2_daily =result_df2['daily returns'].var()
var3_daily =result_df3['daily returns'].var()
var4_daily =result_df4['daily returns'].var()
var5_daily =result_df5['daily returns'].var()
var6_daily =result_df6['daily returns'].var()
var7_daily =result_df7['daily returns'].var()
var8_daily =result_df8['daily returns'].var()
var9_daily =result_df9['daily returns'].var()
var10_daily =result_df10['daily returns'].var()

var1_annual = result_df1['daily returns'].var()*253 #rotn ur 
var2_annual = result_df2['daily returns'].var()*253
var3_annual = result_df3['daily returns'].var()*253
var4_annual = result_df4['daily returns'].var()*253
var5_annual = result_df5['daily returns'].var()*253
var6_annual = result_df6['daily returns'].var()*253
var7_annual = result_df7['daily returns'].var()*253
var8_annual = result_df8['daily returns'].var()*253
var9_annual = result_df9['daily returns'].var()*253
var10_annual = result_df10['daily returns'].var()*253
#%%
#daily and annual variance , assuming 253 business days
std1_daily =result_df1['daily returns'].std()
std2_daily =result_df2['daily returns'].std()
std3_daily =result_df3['daily returns'].std()
std4_daily =result_df4['daily returns'].std()
std5_daily =result_df5['daily returns'].std()
std6_daily =result_df6['daily returns'].std()
std7_daily =result_df7['daily returns'].std()
std8_daily =result_df8['daily returns'].std()
std9_daily =result_df9['daily returns'].std()
std10_daily =result_df10['daily returns'].std()

std1_annual = result_df1['daily returns'].std()*np.sqrt(253) 
std2_annual = result_df2['daily returns'].std()*np.sqrt(253)
std3_annual = result_df3['daily returns'].std()*np.sqrt(253)
std4_annual = result_df4['daily returns'].std()*np.sqrt(253)
std5_annual = result_df5['daily returns'].std()*np.sqrt(253)
std6_annual = result_df6['daily returns'].std()*np.sqrt(253)
std7_annual = result_df7['daily returns'].std()*np.sqrt(253)
std8_annual = result_df8['daily returns'].std()*np.sqrt(253)
std9_annual = result_df9['daily returns'].std()*np.sqrt(253)
std10_annual = result_df10['daily returns'].std()*np.sqrt(253)


#%%
#Ranking 1-7 according to std_dev

if std1_annual < 0.005:
    print('Robur Access Sverige Rank is 1')

if 0.005 <= std1_annual < 0.02:
    print('RAS ranking is 2')

if 0.02 <= std1_annual < 0.05:
    print('RAS is 3')

if 0.05 <= std1_annual < 0.10:
    print('RAS ranking is 4')

if 0.10 <= std1_annual < 0.15:
    print('RAS ranking is 5')

if 0.15 <= std1_annual < 0.25:
    print('RAS ranking is 6')

if 0.25<= std1_annual:
    print('RAS ranking is 7')


##############################################

if std2_annual < 0.005:
    print('Robur Bas Action rank is 1')

if 0.005 <= std2_annual < 0.02:
    print('RBA ranking is 2')

if 0.02 <= std2_annual < 0.05:
    print('RBA is 3')

if 0.05 <= std2_annual < 0.10:
    print('RBA ranking is 4')

if 0.10 <= std2_annual < 0.15:
    print('RBA ranking is 5')

if 0.15 <= std2_annual < 0.25:
    print('RBA ranking is 6')

if 0.25<= std2_annual:
    print('RBA ranking is 7')
##################################################

if std3_annual < 0.005:
    print('Robur Allemansfond Komplett ranking 1')

if 0.005 <= std3_annual < 0.02:
    print('RAK ranking is 2')

if 0.02 <= std3_annual < 0.05:
    print('RAK ranking is 3')

if 0.05 <= std3_annual < 0.10:
    print('RAK ranking is 4')

if 0.10 <= std3_annual < 0.15:
    print('RAK ranking is 5')

if 0.15 <= std3_annual < 0.25:
    print('RAK ranking is 6')

if 0.25<= std3_annual:
    print('RAK ranking is 7')
###########################################

if std4_annual < 0.005:
    print('Robur Amerikafond ranking is 1')

if 0.005 <= std4_annual < 0.02:
    print('RA ranking is 2')

if 0.02 <= std4_annual < 0.05:
    print('RA ranking is 3')

if 0.05 <= std4_annual < 0.10:
    print('RA ranking is 4')

if 0.10 <= std4_annual < 0.15:
    print('RA ranking is 5')

if 0.15 <= std4_annual < 0.25:
    print('RA ranking is 6')

if 0.25<= std4_annual:
    print('RA ranking is 7')
##############################################

if std5_annual < 0.005:
    print('Robur Europafond ranking is 1')

if 0.005 <= std5_annual < 0.02:
    print('RE ranking is 2')

if 0.02 <= std5_annual < 0.05:
    print('RE ranking is 3')

if 0.05 <= std5_annual < 0.10:
    print('RE ranking is 4')

if 0.10 <= std5_annual < 0.15:
    print('RE ranking is 5')

if 0.15 <= std5_annual < 0.25:
    print('RE ranking is 6')

if 0.25<= std5_annual:
    print('RE ranking is 7')
##################################################

if std6_annual < 0.005:
    print('Robur Kapitaltrygg/Dynamic ranking is 1')

if 0.005 <= std6_annual < 0.02:
    print('RK ranking is 2')

if 0.02 <= std6_annual < 0.05:
    print('RK ranking is 3')

if 0.05 <= std6_annual < 0.10:
    print('RK ranking is 4')

if 0.10 <= std6_annual < 0.15:
    print('RK ranking is 5')

if 0.15 <= std6_annual < 0.25:
    print('RK ranking is 6')

if 0.25 <= std6_annual:
    print('RK ranking is 7')
################################################

if std7_annual < 0.005:
    print('Robur Bas Mix ranking is 1')

if 0.005 <= std7_annual < 0.02:
    print('RBM ranking is 2')

if 0.02 <= std7_annual < 0.05:
    print('RBM ranking is 3')

if 0.05 <= std7_annual < 0.10:
    print('RBM ranking is 4')

if 0.10 <= std7_annual < 0.15:
    print('RBM ranking is 5')

if 0.15 <= std7_annual < 0.25:
    print('RBM ranking is 6')

if 0.25 <= std7_annual:
    print('RBM ranking is 7')
#############################################

if std8_annual < 0.005:
    print('Robur Access Mix is 1')

if 0.005 <= std8_annual < 0.02:
    print('RAM ranking is 2')

if 0.02 <= std8_annual < 0.05:
    print('RAM ranking is 3')

if 0.05 <= std8_annual < 0.10:
    print('RAM ranking is 4')

if 0.10 <= std8_annual < 0.15:
    print('RAM ranking is 5')

if 0.15 <= std8_annual < 0.25:
    print('RAM ranking is 6')

if 0.25 <= std8_annual:
    print('RAM ranking is 7')
#################################################

if std9_annual < 0.005:
    print('Robur Asien is 1')

if 0.005 <= std9_annual < 0.02:
    print('RA ranking is 2')

if 0.02 <= std9_annual < 0.05:
    print('RA ranking is 3')

if 0.05 <= std9_annual < 0.10:
    print('RA ranking is 4')

if 0.10 <= std9_annual < 0.15:
    print('RA ranking is 5')

if 0.15 <= std9_annual < 0.25:
    print('RA ranking is 6')

if 0.25 <= std9_annual:
    print('RA ranking is 7')
##########################################

if std10_annual < 0.005:
    print('Robur Bas Solid ranking is 1')

if 0.005 <= std10_annual < 0.02:
    print('RBS ranking is 2')

if 0.02 <= std10_annual < 0.05:
    print('RBS ranking is 3')

if 0.05 <= std10_annual < 0.10:
    print('RBS ranking is 4')

if 0.10 <= std10_annual < 0.15:
    print('RBS ranking is 5')

if 0.15 <= std10_annual < 0.25:
    print('RBS ranking is 6')

if 0.25 <= std10_annual:
    print('RBS ranking is 7')
#########################################################
#%%
# Sample period SP
sp1 = len(result_df1)
sp2 = len(result_df2)
sp3 = len(result_df3)
sp4 = len(result_df4)
sp5 = len(result_df5)
sp6 = len(result_df6)
sp7 = len(result_df7)
sp8 = len(result_df8)
sp9 = len(result_df9)
sp10 = len(result_df10)

#Recommended holding period RHP (entered manually depending on fund)
rhp1=1261
rhp2=1261
rhp3=1261
rhp4=1261
rhp5=1261
rhp6=754
rhp7=754
rhp8=754
rhp9=1261
rhp10=754

#%%
rhp_sp1 = rhp1/sp1
rhp_sp2 = rhp2/sp2
rhp_sp3 = rhp3/sp3
rhp_sp4 = rhp4/sp4
rhp_sp5 = rhp5/sp5
rhp_sp6 = rhp6/sp6
rhp_sp7 = rhp7/sp7
rhp_sp8 = rhp8/sp8
rhp_sp9 = rhp9/sp9
rhp_sp10 = rhp10/sp10

#%%
# Weighted estimater WE
we1 = (arithmeticMean1 * (1-rhp_sp1))+(geometricMean1*rhp_sp1)
we2 = (arithmeticMean2 * (1-rhp_sp2))+(geometricMean2*rhp_sp2)
we3 = (arithmeticMean3 * (1-rhp_sp3))+(geometricMean3*rhp_sp3)
we4 = (arithmeticMean4 * (1-rhp_sp4))+(geometricMean4*rhp_sp4)
we5 = (arithmeticMean5 * (1-rhp_sp5))+(geometricMean5*rhp_sp5)
we6 = (arithmeticMean6 * (1-rhp_sp6))+(geometricMean6*rhp_sp6)
we7 = (arithmeticMean7 * (1-rhp_sp7))+(geometricMean7*rhp_sp7)
we8 = (arithmeticMean8 * (1-rhp_sp8))+(geometricMean8*rhp_sp8)
we9 = (arithmeticMean9 * (1-rhp_sp9))+(geometricMean9*rhp_sp9)
we10 = (arithmeticMean10 * (1-rhp_sp10))+(geometricMean10*rhp_sp10)
 #%%
 #GM-WE test
gm_we1 = we1-geometricMean1
gm_we2 = we2-geometricMean2
gm_we3 = we3-geometricMean3
gm_we4 = we4-geometricMean4
gm_we5 = we5-geometricMean5
gm_we6 = we6-geometricMean6
gm_we7 = we7-geometricMean7
gm_we8 = we8-geometricMean8
gm_we9 = we9-geometricMean9
gm_we10 = we10-geometricMean10


#%%
# Value at Risk (95 confidence)
mean1=np.mean(rets1)
std_dev1=np.std(rets1)
VaR1_95 = norm.ppf(1-0.95, mean1, std_dev1)
print(VaR1_95)

mean2=np.mean(rets2)
std_dev2=np.std(rets2)
VaR2_95 = norm.ppf(1-0.95, mean2, std_dev2)
print(VaR2_95)

mean3=np.mean(rets3)
std_dev3=np.std(rets3)
VaR3_95 = norm.ppf(1-0.95, mean3, std_dev3)
print(VaR3_95)

mean4=np.mean(rets4)
std_dev4=np.std(rets4)
VaR4_95 = norm.ppf(1-0.95, mean4, std_dev4)
print(VaR4_95)

mean5=np.mean(rets5)
std_dev5=np.std(rets5)
VaR5_95 = norm.ppf(1-0.95, mean5, std_dev5)
print(VaR5_95)

mean6=np.mean(rets6)
std_dev6=np.std(rets6)
VaR6_95 = norm.ppf(1-0.95, mean6, std_dev6)
print(VaR6_95)

mean7=np.mean(rets7)
std_dev7=np.std(rets7)
VaR7_95 = norm.ppf(1-0.95, mean7, std_dev7)
print(VaR7_95)

mean8=np.mean(rets8)
std_dev8=np.std(rets8)
VaR8_95 = norm.ppf(1-0.95, mean8, std_dev8)
print(VaR8_95)

mean9=np.mean(rets9)
std_dev9=np.std(rets9)
VaR9_95 = norm.ppf(1-0.95, mean9, std_dev9)
print(VaR9_95)

mean10=np.mean(rets10)
std_dev10=np.std(rets10)
VaR10_95 = norm.ppf(1-0.95, mean10, std_dev10)
print(VaR10_95)

#%%
# Annual VaR(95)
annual_VaR1_95 = VaR1_95*np.sqrt(253*0.05)
annual_VaR2_95 = VaR2_95*np.sqrt(253*0.05)
annual_VaR3_95 = VaR3_95*np.sqrt(253*0.05)
annual_VaR4_95 = VaR4_95*np.sqrt(253*0.05)
annual_VaR5_95 = VaR5_95*np.sqrt(253*0.05)
annual_VaR6_95 = VaR6_95*np.sqrt(253*0.05)
annual_VaR7_95 = VaR7_95*np.sqrt(253*0.05)
annual_VaR8_95 = VaR8_95*np.sqrt(253*0.05)
annual_VaR9_95 = VaR9_95*np.sqrt(253*0.05)
annual_VaR10_95 = VaR10_95*np.sqrt(253*0.05)

#%%
# CVaR(95)
alpha = 0.05
CVaR1_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev1 - mean1
print("95% CVaR/ES is", round(CVaR1_95*100,2))

CVaR2_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev2 - mean2
print("95% CVaR/ES is", round(CVaR2_95*100,2))

CVaR3_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev3 - mean3
print("95% CVaR/ES is", round(CVaR3_95*100,2))

CVaR4_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev4 - mean4
print("95% CVaR/ES is", round(CVaR4_95*100,2))

CVaR5_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev5 - mean5
print("95% CVaR/ES is", round(CVaR5_95*100,2))

CVaR6_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev6 - mean6
print("95% CVaR/ES is", round(CVaR6_95*100,2))

CVaR7_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev7 - mean7
print("95% CVaR/ES is", round(CVaR7_95*100,2))

CVaR8_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev8 - mean8
print("95% CVaR/ES is", round(CVaR8_95*100,2))

CVaR9_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev9 - mean9
print("95% CVaR/ES is", round(CVaR9_95*100,2))

CVaR10_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev10 - mean10
print("95% CVaR/ES is", round(CVaR10_95*100,2))

#%%
# Annual CVaR(95)
annual_CVaR1_95 = CVaR1_95*np.sqrt(253*0.05)
annual_CVaR2_95 = CVaR2_95*np.sqrt(253*0.05)
annual_CVaR3_95 = CVaR3_95*np.sqrt(253*0.05)
annual_CVaR4_95 = CVaR4_95*np.sqrt(253*0.05)
annual_CVaR5_95 = CVaR5_95*np.sqrt(253*0.05)
annual_CVaR6_95 = CVaR6_95*np.sqrt(253*0.05)
annual_CVaR7_95 = CVaR7_95*np.sqrt(253*0.05)
annual_CVaR8_95 = CVaR8_95*np.sqrt(253*0.05)
annual_CVaR9_95 = CVaR9_95*np.sqrt(253*0.05)
annual_CVaR10_95 = CVaR10_95*np.sqrt(253*0.05)

#%%
#correlation matrix (dataframe type)
df_pivot=df.pivot('date','fund index','daily returns').drop(np.nan,axis=1).reset_index()
corr_df = df_pivot.corr(method='pearson')
corr_df1=corr_df.drop(['date']).drop(['date'],axis=1)
#%%
#sqrd StDev matrix
std= [std1_daily,std2_daily,std3_daily,std4_daily,std5_daily,std6_daily,std7_daily,std8_daily,std9_daily,std10_daily]
std_array=np.array([std])
std_array2=std_array.reshape((-1,1))
sqrd_stdev=np.dot(std_array2,std_array) 

#%%
#Var/Covar matrix
var_cov=np.multiply(corr_df1,sqrd_stdev)
         
#%%
#Variance Identity matrix 
variance_m = corr_df1 #starting) with correlation matrix
# i= row, j=column, i=j=1 and otherwise 0 
v_i = variance_m==1
v_i[v_i == True]=1
print(v_i)
#%% 
#Hollow Logical matrix 
variance_m = corr_df1 #starting with correlation matrix
# i= row, j=column, i=j=0 and otherwise 1
H_l = variance_m != 1
H_l[H_l == True]=1
print(H_l)
#%%
#Variance matrix
variance_matrix= (v_i*var_cov)
print(variance_matrix)
#%%
cov_matrix=(H_l*var_cov)
print(cov_matrix)
#%%
#Daily returns dataframe for calculating historical VaR and CVaR
s=pd.DataFrame(np.sort(df_pivot,axis=0))
s1=s.drop([0],inplace=True,axis=1) #drop date column
#%%
count=s.count()
print(count)
#%%
#99.9
ninty_nine_nine1= math.ceil(count[1]*0.001)
ninty_nine_nine2= math.ceil(count[2]*0.001)
ninty_nine_nine3= math.ceil(count[3]*0.001)
ninty_nine_nine4= math.ceil(count[4]*0.001)
ninty_nine_nine5= math.ceil(count[5]*0.001)
ninty_nine_nine6= math.ceil(count[6]*0.001)
ninty_nine_nine7= math.ceil(count[7]*0.001)
ninty_nine_nine8= math.ceil(count[8]*0.001)
ninty_nine_nine9= math.ceil(count[9]*0.001)
ninty_nine_nine10= math.ceil(count[10]*0.001)
#%%
#99
ninty_nine1=math.ceil(count[1]*0.01)
ninty_nine2=math.ceil(count[2]*0.01)
ninty_nine3=math.ceil(count[3]*0.01)
ninty_nine4=math.ceil(count[4]*0.01)
ninty_nine5=math.ceil(count[5]*0.01)
ninty_nine6=math.ceil(count[6]*0.01)
ninty_nine7=math.ceil(count[7]*0.01)
ninty_nine8=math.ceil(count[8]*0.01)
ninty_nine9=math.ceil(count[9]*0.01)
ninty_nine10=math.ceil(count[10]*0.01)
#%%
#95
ninty_five1=math.ceil(count[1]*0.05)
ninty_five2=math.ceil(count[2]*0.05)
ninty_five3=math.ceil(count[3]*0.05)
ninty_five4=math.ceil(count[4]*0.05)
ninty_five5=math.ceil(count[5]*0.05)
ninty_five6=math.ceil(count[6]*0.05)
ninty_five7=math.ceil(count[7]*0.05)
ninty_five8=math.ceil(count[8]*0.05)
ninty_five9=math.ceil(count[9]*0.05)
ninty_five10=math.ceil(count[10]*0.05)
#%%
#Reference Cell
Rc1=s.loc[ninty_five1,1]
Rc2=s.loc[ninty_five2,2]
Rc3=s.loc[ninty_five3,3]
Rc4=s.loc[ninty_five4,4]
Rc5=s.loc[ninty_five5,5]
Rc6=s.loc[ninty_five6,6]
Rc7=s.loc[ninty_five7,7]
Rc8=s.loc[ninty_five8,8]
Rc9=s.loc[ninty_five9,9]
Rc10=s.loc[ninty_five10,10]
#%%
#Var(95) in percentage
var95_1="{:.4%}".format(VaR1_95)
var95_2="{:.4%}".format(VaR2_95)
var95_3="{:.4%}".format(VaR3_95)
var95_4="{:.4%}".format(VaR4_95)
var95_5="{:.4%}".format(VaR5_95)
var95_6="{:.4%}".format(VaR6_95)
var95_7="{:.4%}".format(VaR7_95)
var95_8="{:.4%}".format(VaR8_95)
var95_9="{:.4%}".format(VaR9_95)
var95_10="{:.4%}".format(VaR10_95)

var95_annual1="{:.4%}".format(annual_VaR1_95)
var95_annual2="{:.4%}".format(annual_VaR2_95)
var95_annual3="{:.4%}".format(annual_VaR3_95)
var95_annual4="{:.4%}".format(annual_VaR4_95)
var95_annual5="{:.4%}".format(annual_VaR5_95)
var95_annual6="{:.4%}".format(annual_VaR6_95)
var95_annual7="{:.4%}".format(annual_VaR7_95)
var95_annual8="{:.4%}".format(annual_VaR8_95)
var95_annual9="{:.4%}".format(annual_VaR9_95)
var95_annual10="{:.4%}".format(annual_VaR10_95)

#%%
sumup1=s.where(s[1]>Rc1).sum(0)
sumup2=s.where(s[1]>=0).sum(0)
times1=s.where(s[1]>Rc1).count(0)
times2=s.where(s[1]>=0).count(0)
avg_loss=(sumup1-sumup2)/(times1-times2)
#%%
#Avg Loss > VaR(95)  {1}
Rc= np.array([Rc1,Rc2,Rc3,Rc4,Rc5,Rc6,Rc7,Rc8,Rc9,Rc10])
m,n=s.shape
Sum1=np.zeros(n)
count1=np.zeros(n)
sample=s.to_numpy()
for i in range(n):
    for j in range(m):
        if sample[j,i] > Rc[i] :
            Sum1[i] = Sum1[i]+sample[j,i]
            count1[i] +=1
#print(Sum1)
#print(count1)
#%%
#Avg Loss > VaR(95) {2}
Sum2=np.zeros(n)
count2=np.zeros(n)
sample=s.to_numpy()
for i in range(n):
    for j in range(m):
        if sample[j,i] >= 0 :
           Sum2[i] = Sum2[i]+sample[j,i]
           count2[i] +=1
print(Sum2)
print(count2)
#%%
#Avg Loss > VaR(95)

Avg_Loss = geek.subtract(Sum1,Sum2)/geek.subtract(count1,count2)
print(Avg_Loss)
#%%
#sqrt CVaR
sqrtCVaR1 = np.sqrt(CVaR1_95)
sqrtCVaR2 = np.sqrt(CVaR2_95)
sqrtCVaR3 = np.sqrt(CVaR3_95)
sqrtCVaR4 = np.sqrt(CVaR4_95)
sqrtCVaR5 = np.sqrt(CVaR5_95)
sqrtCVaR6 = np.sqrt(CVaR6_95)
sqrtCVaR7 = np.sqrt(CVaR7_95)
sqrtCVaR8 = np.sqrt(CVaR8_95)
sqrtCVaR9 = np.sqrt(CVaR9_95)
sqrtCVaR10 = np.sqrt(CVaR10_95)
#%%
#CVaR matrix
sqrtCVaR= [sqrtCVaR1,sqrtCVaR2,sqrtCVaR3,sqrtCVaR4,sqrtCVaR5,sqrtCVaR6,sqrtCVaR7,sqrtCVaR8,sqrtCVaR9,sqrtCVaR10]
sqrtCVaR_array=np.array([sqrtCVaR])
sqrtCVaR_array2=sqrtCVaR_array.reshape((-1,1))
CVaR_matrix=np.dot(sqrtCVaR_array2,sqrtCVaR_array)
#%%
#Avg non-CVaR loss matrix

#%%
#CoCVaR matrix (AvgRet>VaR(95))


VaR= [VaR1_95,VaR2_95,VaR3_95,VaR4_95,VaR5_95,VaR6_95,VaR7_95,VaR8_95,VaR9_95,VaR10_95,]
VaR_array=np.array([VaR])
VaR_array1=np.multiply(VaR_array,-1)
VaR_array2=VaR_array1.reshape((-1,1))

m,n = corr_df1.shape
Sum2=np.zeros(n)
count2=np.zeros(n)
corr_df2=corr_df1.to_numpy()
CoCVaR=np.zeros((m,n))

check1=corr_df2*VaR_array2

for i in range(n):
    for j in range(m):
        if (corr_df2[j,i] * VaR_array2[i]) < VaR_array1[0,j]:
            CoCVaR[j,i] = 1 
        else: 
            CoCVaR[j,i]= 0

 #%%
#CoCVaR  
    
CoCVaR4= (CVaR_matrix*v_i)

#%%
#Avg Ret > VaR(95)
Avg_Ret=np.multiply(avg_loss,-1)

#%%
#Avg non-CVaR Loss matrix
Avg_Ret2=Avg_Ret.to_numpy()
Avg_Ret1=Avg_Ret2.reshape((-1,1))


non_CVaR=np.zeros(shape=(m,n))
non_CVaR1=np.mat(Avg_Ret1)
for i in range(n):
    for j in range(m):
        non_CVaR[i]=Avg_Ret1[i]

#%%
#CoCVaR Avg Ret>Var(95),var(95)
CoCVaR_Avg_Ret=np.multiply(non_CVaR,CoCVaR)

#%%
#CoCVar Matrix (AvgRet<VaR95)
CoCVaR_Avg_less_Ret=np.multiply(CoCVaR_Avg_Ret,corr_df1)



####################################################################



#%%
#Optimizer

#Individual Returns (Weighted estimater WE)
we1 = (arithmeticMean1 * (1-rhp_sp1))+(geometricMean1*rhp_sp1)
we2 = (arithmeticMean2 * (1-rhp_sp2))+(geometricMean2*rhp_sp2)
we3 = (arithmeticMean3 * (1-rhp_sp3))+(geometricMean3*rhp_sp3)
we4 = (arithmeticMean4 * (1-rhp_sp4))+(geometricMean4*rhp_sp4)
we5 = (arithmeticMean5 * (1-rhp_sp5))+(geometricMean5*rhp_sp5)
we6 = (arithmeticMean6 * (1-rhp_sp6))+(geometricMean6*rhp_sp6)
we7 = (arithmeticMean7 * (1-rhp_sp7))+(geometricMean7*rhp_sp7)
we8 = (arithmeticMean8 * (1-rhp_sp8))+(geometricMean8*rhp_sp8)
we9 = (arithmeticMean9 * (1-rhp_sp9))+(geometricMean9*rhp_sp9)
we10 = (arithmeticMean10 * (1-rhp_sp10))+(geometricMean10*rhp_sp10)

#Individual StD (assuming 253 business days)
std1_daily =result_df1['daily returns'].std()
std2_daily =result_df2['daily returns'].std()
std3_daily =result_df3['daily returns'].std()
std4_daily =result_df4['daily returns'].std()
std5_daily =result_df5['daily returns'].std()
std6_daily =result_df6['daily returns'].std()
std7_daily =result_df7['daily returns'].std()
std8_daily =result_df8['daily returns'].std()
std9_daily =result_df9['daily returns'].std()
std10_daily =result_df10['daily returns'].std()

#CVaR(95)
alpha = 0.05
CVaR1_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev1 - mean1
print("95% CVaR/ES is", round(CVaR1_95*100,2))

CVaR2_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev2 - mean2
print("95% CVaR/ES is", round(CVaR2_95*100,2))

CVaR3_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev3 - mean3
print("95% CVaR/ES is", round(CVaR3_95*100,2))

CVaR4_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev4 - mean4
print("95% CVaR/ES is", round(CVaR4_95*100,2))

CVaR5_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev5 - mean5
print("95% CVaR/ES is", round(CVaR5_95*100,2))

CVaR6_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev6 - mean6
print("95% CVaR/ES is", round(CVaR6_95*100,2))

CVaR7_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev7 - mean7
print("95% CVaR/ES is", round(CVaR7_95*100,2))

CVaR8_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev8 - mean8
print("95% CVaR/ES is", round(CVaR8_95*100,2))

CVaR9_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev9 - mean9
print("95% CVaR/ES is", round(CVaR9_95*100,2))

CVaR10_95 = alpha**-1 * norm.pdf(norm.ppf(alpha))*std_dev10 - mean10
print("95% CVaR/ES is", round(CVaR10_95*100,2))

#Individual rating
#%%
#Ranking 1-7 according to std_dev

if std1_annual < 0.005:
    print('Robur Access Sverige Rank is 1')

if 0.005 <= std1_annual < 0.02:
    print('RAS ranking is 2')

if 0.02 <= std1_annual < 0.05:
    print('RAS is 3')

if 0.05 <= std1_annual < 0.10:
    print('RAS ranking is 4')

if 0.10 <= std1_annual < 0.15:
    print('RAS ranking is 5')

if 0.15 <= std1_annual < 0.25:
    print('RAS ranking is 6')

if 0.25<= std1_annual:
    print('RAS ranking is 7')


##############################################

if std2_annual < 0.005:
    print('Robur Bas Action rank is 1')

if 0.005 <= std2_annual < 0.02:
    print('RBA ranking is 2')

if 0.02 <= std2_annual < 0.05:
    print('RBA is 3')

if 0.05 <= std2_annual < 0.10:
    print('RBA ranking is 4')

if 0.10 <= std2_annual < 0.15:
    print('RBA ranking is 5')

if 0.15 <= std2_annual < 0.25:
    print('RBA ranking is 6')

if 0.25<= std2_annual:
    print('RBA ranking is 7')
##################################################

if std3_annual < 0.005:
    print('Robur Allemansfond Komplett ranking 1')

if 0.005 <= std3_annual < 0.02:
    print('RAK ranking is 2')

if 0.02 <= std3_annual < 0.05:
    print('RAK ranking is 3')

if 0.05 <= std3_annual < 0.10:
    print('RAK ranking is 4')

if 0.10 <= std3_annual < 0.15:
    print('RAK ranking is 5')

if 0.15 <= std3_annual < 0.25:
    print('RAK ranking is 6')

if 0.25<= std3_annual:
    print('RAK ranking is 7')
###########################################

if std4_annual < 0.005:
    print('Robur Amerikafond ranking is 1')

if 0.005 <= std4_annual < 0.02:
    print('RA ranking is 2')

if 0.02 <= std4_annual < 0.05:
    print('RA ranking is 3')

if 0.05 <= std4_annual < 0.10:
    print('RA ranking is 4')

if 0.10 <= std4_annual < 0.15:
    print('RA ranking is 5')

if 0.15 <= std4_annual < 0.25:
    print('RA ranking is 6')

if 0.25<= std4_annual:
    print('RA ranking is 7')
##############################################

if std5_annual < 0.005:
    print('Robur Europafond ranking is 1')

if 0.005 <= std5_annual < 0.02:
    print('RE ranking is 2')

if 0.02 <= std5_annual < 0.05:
    print('RE ranking is 3')

if 0.05 <= std5_annual < 0.10:
    print('RE ranking is 4')

if 0.10 <= std5_annual < 0.15:
    print('RE ranking is 5')

if 0.15 <= std5_annual < 0.25:
    print('RE ranking is 6')

if 0.25<= std5_annual:
    print('RE ranking is 7')
##################################################

if std6_annual < 0.005:
    print('Robur Kapitaltrygg/Dynamic ranking is 1')

if 0.005 <= std6_annual < 0.02:
    print('RK ranking is 2')

if 0.02 <= std6_annual < 0.05:
    print('RK ranking is 3')

if 0.05 <= std6_annual < 0.10:
    print('RK ranking is 4')

if 0.10 <= std6_annual < 0.15:
    print('RK ranking is 5')

if 0.15 <= std6_annual < 0.25:
    print('RK ranking is 6')

if 0.25 <= std6_annual:
    print('RK ranking is 7')
################################################

if std7_annual < 0.005:
    print('Robur Bas Mix ranking is 1')

if 0.005 <= std7_annual < 0.02:
    print('RBM ranking is 2')

if 0.02 <= std7_annual < 0.05:
    print('RBM ranking is 3')

if 0.05 <= std7_annual < 0.10:
    print('RBM ranking is 4')

if 0.10 <= std7_annual < 0.15:
    print('RBM ranking is 5')

if 0.15 <= std7_annual < 0.25:
    print('RBM ranking is 6')

if 0.25 <= std7_annual:
    print('RBM ranking is 7')
#############################################

if std8_annual < 0.005:
    print('Robur Access Mix is 1')

if 0.005 <= std8_annual < 0.02:
    print('RAM ranking is 2')

if 0.02 <= std8_annual < 0.05:
    print('RAM ranking is 3')

if 0.05 <= std8_annual < 0.10:
    print('RAM ranking is 4')

if 0.10 <= std8_annual < 0.15:
    print('RAM ranking is 5')

if 0.15 <= std8_annual < 0.25:
    print('RAM ranking is 6')

if 0.25 <= std8_annual:
    print('RAM ranking is 7')
#################################################

if std9_annual < 0.005:
    print('Robur Asien is 1')

if 0.005 <= std9_annual < 0.02:
    print('RA ranking is 2')

if 0.02 <= std9_annual < 0.05:
    print('RA ranking is 3')

if 0.05 <= std9_annual < 0.10:
    print('RA ranking is 4')

if 0.10 <= std9_annual < 0.15:
    print('RA ranking is 5')

if 0.15 <= std9_annual < 0.25:
    print('RA ranking is 6')

if 0.25 <= std9_annual:
    print('RA ranking is 7')
##########################################

if std10_annual < 0.005:
    print('Robur Bas Solid ranking is 1')

if 0.005 <= std10_annual < 0.02:
    print('RBS ranking is 2')

if 0.02 <= std10_annual < 0.05:
    print('RBS ranking is 3')

if 0.05 <= std10_annual < 0.10:
    print('RBS ranking is 4')

if 0.10 <= std10_annual < 0.15:
    print('RBS ranking is 5')

if 0.15 <= std10_annual < 0.25:
    print('RBS ranking is 6')

if 0.25 <= std10_annual:
    print('RBS ranking is 7')
#########################################################
#%%
#Annualized returns (assuming 253 business 
ann_Ret1=((1+we1)**253)-1
ann_Ret2=((1+we2)**253)-1
ann_Ret3=((1+we3)**253)-1
ann_Ret4=((1+we4)**253)-1
ann_Ret5=((1+we5)**253)-1
ann_Ret6=((1+we6)**253)-1
ann_Ret7=((1+we7)**253)-1
ann_Ret8=((1+we8)**253)-1
ann_Ret9=((1+we9)**253)-1
ann_Ret10=((1+we10)**253)-1
#%%
# Annualized StD (assuming 253 business days)
std1_annual = std1_daily*np.sqrt(253) 
std2_annual = std2_daily*np.sqrt(253) 
std3_annual = std3_daily*np.sqrt(253) 
std4_annual = std4_daily*np.sqrt(253) 
std5_annual = std5_daily*np.sqrt(253) 
std6_annual = std6_daily*np.sqrt(253) 
std7_annual = std7_daily*np.sqrt(253) 
std8_annual = std8_daily*np.sqrt(253) 
std9_annual = std9_daily*np.sqrt(253)
std10_annual = std10_daily*np.sqrt(253)  

#Annual CVaR(95)
annual_CVaR1_95 = CVaR1_95*np.sqrt(253*0.05)
annual_CVaR2_95 = CVaR2_95*np.sqrt(253*0.05)
annual_CVaR3_95 = CVaR3_95*np.sqrt(253*0.05)
annual_CVaR4_95 = CVaR4_95*np.sqrt(253*0.05)
annual_CVaR5_95 = CVaR5_95*np.sqrt(253*0.05)
annual_CVaR6_95 = CVaR6_95*np.sqrt(253*0.05)
annual_CVaR7_95 = CVaR7_95*np.sqrt(253*0.05)
annual_CVaR8_95 = CVaR8_95*np.sqrt(253*0.05)
annual_CVaR9_95 = CVaR9_95*np.sqrt(253*0.05)
annual_CVaR10_95 = CVaR10_95*np.sqrt(253*0.05)
#%%
#Port Total VaR
weight=np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])#will change depending on opt model
port_total_var=np.dot(weight,np.matmul(var_cov,weight.T))
#%%
#Port StD
port_std=np.sqrt(port_total_var)
#%%
#Port Var
port_var=np.dot(weight,(np.matmul(weight,variance_matrix).T))
#%%
#Port Cov
port_cov=np.dot(weight,(np.matmul(weight,cov_matrix).T))
#%%
#Port_ret
ind_ret=np.array([we1,we2,we3,we4,we5,we6,we7,we8,we9,we10])#.reshape(1,-1)
port_ret= np.sum(weight*ind_ret)
#%%
#Port CVaR(95)
cvar_95=np.array([CVaR1_95,CVaR2_95,CVaR3_95,CVaR4_95,CVaR5_95,CVaR6_95,CVaR7_95,CVaR8_95,CVaR9_95,CVaR10_95,]).reshape(1,-1)
port_cvar=np.sum(weight*cvar_95)
#%%
#Port CoCVaR(95)
port_cocvar=(np.dot(weight,(np.dot(weight,CoCVaR_Avg_Ret)).T))*2
#%%
#Optimization options

#Annualized Tot StD
ann_tot_std=(port_std*np.sqrt(253))
#%%
#Portfolio rating 
if ann_tot_std < 0.005:
    print('Portfolio rating is 1')

if 0.005 <= ann_tot_std < 0.02:
    print('Portfolio rating is 2')

if 0.02 <= ann_tot_std < 0.05:
    print('Portfolio rating is 3')

if 0.05 <= ann_tot_std < 0.10:
    print('Portfolio rating  is 4')

if 0.10 <= ann_tot_std< 0.15:
    print('Portfolio rating  is 5')

if 0.15 <= ann_tot_std < 0.25:
    print('Portfolio rating  is 6')

if 0.25 <= ann_tot_std:
    print('Portfolio rating is 7')
#%%
#Annualized Tot Ret
ann_tot_ret=(((1+port_ret)**253)-1)
#%%
#Annualized Ret/Vol , sharpe ratio
ann_ret_vol=(ann_tot_ret/(ann_tot_std+0.00000000000001))
#%%
#Annualized CVaR(95)
ann_cvar=((port_cvar+port_cocvar)*np.sqrt(253*0.05))
#%%
#Annualized Ret/(Vol+CVaR)
ann_ret_vol_cvar=ann_tot_ret/(ann_cvar+ann_tot_std+0.00000000001)
#%%
#Annualized Ret/CVaR
ann_ret_cvar=ann_tot_ret/(ann_cvar+0.000000001)
#%%
#RHP
rhp=np.array([rhp1,rhp2,rhp3,rhp4,rhp5,rhp6,rhp7,rhp8,rhp9,rhp10])
port_rhp=round(np.sum(weight*rhp)/253)   

#%%
#Optimization model

#objective function
#right side needs to be 0
#minimize function so we want to maximize we want to minimize the negative objective
 
x = weight
#def objective(x): #minimize risk
    #port_total_var=np.dot(x,np.matmul(var_cov,x.T))
    #port_std=np.sqrt(port_total_var)
    #return (port_std*np.sqrt(253))
def objective(x): #maximize return 
    return -(((1+(np.dot(x,ind_ret)))**253)-1)
#constraints 
def constraint1(x): #weights sum up to 1
    return np.sum(x)-1

def constraint2(x): #diversified portfolio, max weight=20%
    return 1/5-min(x)

def constraint3(x): #minimum weight 0, no short-selling
    return 0-min(x) 

rl=0.10
def constraint4(x): #constraint for the risk level, dont worry about it in this step (for the ef plot)
    port_total_var=np.dot(x,np.matmul(var_cov,x.T))
    port_std=np.sqrt(port_total_var)
    return rl-(port_std*np.sqrt(253))

def constraint5(x): #constraint for the risk level, dont worry about it in this step (for the ef plot)
    port_cvar=np.sum(x*cvar_95)
    port_cocvar=(np.dot(x,(np.dot(x,CoCVaR_Avg_Ret)).T))*2
    return rl-((port_cvar+port_cocvar)*np.sqrt(253*0.05))


#initial guess
nof=10 #number of funds
initialGuess=np.ones(nof)*(1./nof)
#initialGuess=np.array([0,0,0,0,0,0,0,1,0,0])
print('Total return:',ann_tot_ret)
print('Total risk:',ann_tot_std)
print('Total CVaR:',ann_cvar)
print('Sharpe Ratio:',ann_ret_vol)
print('Initial guess: ',initialGuess)

#optimize
bnds=tuple((0,1)for x in range(nof))
con1={'type':'eq','fun':constraint1}
con2={'type':'eq','fun':constraint2}
con3={'type':'ineq','fun':constraint3}
con4={'type':'ineq','fun':constraint4}
con5={'type':'ineq','fun':constraint5}
cons=([con1,con2,con3,con4])


solution=minimize(objective,initialGuess,method='SLSQP',bounds=bnds,constraints=cons)
Port_results=np.zeros((4,1))
Port_results[0]=(((1+np.dot(solution.x,ind_ret))**253)-1)
Port_results[1]=np.sqrt(np.dot(solution.x,np.matmul(var_cov,solution.x.T)))*np.sqrt(253)
Port_results[2]=(((np.sum(solution.x*cvar_95))+((np.dot(weight,(np.dot(weight,CoCVaR_Avg_Ret)).T))*2))*np.sqrt(253*0.05))
Port_results[3]=((Port_results[0])/((Port_results[2])+0.00000000000001))
print('Total return after optimizer',Port_results[0]) #annual return after optimizer

print("Total risk after optimzer",Port_results[1])
print('Total CVaR after optimizer',Port_results[2])
print('Sharpe Ratio after optimizer',Port_results[3])
print('Weights for portfolio with maximum returns',solution['x'].round(4))
#%%
n=50
rl=np.linspace(0.00728871,0.16407658,n)

#rl=np.linspace(0.035,0.16,n)
#rl=rl[::-1]
Port_results=np.zeros((2,n))
for i in range(0,n):
    x = weight
    def objective(x): #maximize return 
        return -(((1+(np.dot(x,ind_ret)))**253)-1)
    
    def constraint1(x): #weights sum up to 1
        return np.sum(x)-1
    
    def constraint2(x): #diversified portfolio, max weight=20%
        return 1/5-max(x)
    
    def constraint3(x): #minimum weight 0, no short-selling
        return 0-min(x) 
    
    def constraint4(x): #boundary points for the risklevel,used when plotting EF
        port_total_var=np.dot(x,np.matmul(var_cov,x.T))
        port_std=np.sqrt(port_total_var)
        return rl[i]-(port_std*np.sqrt(253))
    
    nof=10 #number of funds
    initialGuess=np.ones(nof)*(1./nof)
    #optimize
    bnds=tuple((0,1)for x in range(nof))
    con1={'type':'eq','fun':constraint1}
    con2={'type':'eq','fun':constraint2}
    con3={'type':'ineq','fun':constraint3}
    con4={'type':'ineq','fun':constraint4}
    cons=([con1,con4])

    
    
    solution=minimize(objective,initialGuess,method='SLSQP',bounds=bnds,constraints=cons)

    Port_results[0,i]=(((1+np.dot(solution.x,ind_ret))**253)-1)
    Port_results[1,i]=np.sqrt(np.dot(solution.x,np.matmul(var_cov,solution.x.T)))*np.sqrt(253)
    
    #print(i,Port_results[0,i],Port_results[1,i])
    #print(i,rl[i],Port_results[0,i],Port_results[1,i])
#%%
port_total_var1=np.dot(solution.x,np.matmul(var_cov,solution.x.T))
#%%
#Port StD
port_std1=np.sqrt(port_total_var1)
#%%
#Port Var
port_var1=np.dot(solution.x,(np.matmul(solution.x,variance_matrix).T))
#%%
#Port Cov
port_cov1=np.dot(solution.x,(np.matmul(solution.x,cov_matrix).T))
#%%
#Port_ret
ind_ret1=np.array([we1,we2,we3,we4,we5,we6,we7,we8,we9,we10]).reshape(1,-1)
port_ret1= np.sum(solution.x*ind_ret1)
#%%
#Port CVaR(95)
cvar_951=np.array([CVaR1_95,CVaR2_95,CVaR3_95,CVaR4_95,CVaR5_95,CVaR6_95,CVaR7_95,CVaR8_95,CVaR9_95,CVaR10_95,]).reshape(1,-1)
port_cvar1=np.sum(solution.x*cvar_951)
#%%
#Port CoCVaR(95)
port_cocvar1=(np.dot(solution.x,(np.dot(solution.x,CoCVaR_Avg_Ret)).T))*2
#%%
#Annualized Tot StD
ann_tot_std1=(port_std1*np.sqrt(253))
#%%
#Portfolio rating 
if ann_tot_std1 < 0.005:
    print('Portfolio rating is 1')

if 0.005 <= ann_tot_std1 < 0.02:
    print('Portfolio rating is 2')

if 0.02 <= ann_tot_std1 < 0.05:
    print('Portfolio rating is 3')

if 0.05 <= ann_tot_std1 < 0.10:
    print('Portfolio rating  is 4')

if 0.10 <= ann_tot_std1< 0.15:
    print('Portfolio rating  is 5')

if 0.15 <= ann_tot_std1 < 0.25:
    print('Portfolio rating  is 6')

if 0.25 <= ann_tot_std1:
    print('Portfolio rating is 7')
#%%
#Annualized Tot Ret
ann_tot_ret1=(((1+port_ret1)**253)-1)

#%%
#Annualized Ret/Vol , sharpe ratio
ann_ret_vol1=(ann_tot_ret1/(ann_tot_std1+0.00000000000001))
#%%
#Annualized CVaR(95)
ann_cvar1=((port_cvar1+port_cocvar1)*np.sqrt(253*0.05))
#%%
#Annualized Ret/(Vol+CVaR)
ann_ret_vol_cvar1=ann_tot_ret1/(ann_cvar1+ann_tot_std1+0.00000000001)
#%%
#Annualized Ret/CVaR
ann_ret_cvar1=ann_tot_ret1/(ann_cvar1+0.000000001)
#%%
#RHP
port_rhp1=round(np.sum(solution.x*rhp)/253)   
#%%
plt.figure(dpi=200)
plt.plot(Port_results[1,:],Port_results[0,:],label='Efficent Frontier');#,'o',color='black');
plt.scatter(Port_results[0,i],Port_results[1,i],marker="*",label='OP')
plt.scatter(ann_tot_std,ann_tot_ret,marker="*",label='EW')
plt.legend()
plt.grid(True)

#%%
result_df1['daily returns'].plot()
result_df2['daily returns'].plot()
result_df3['daily returns'].plot()
result_df4['daily returns'].plot()
result_df5['daily returns'].plot()
result_df6['daily returns'].plot()
result_df7['daily returns'].plot()
result_df8['daily returns'].plot()
result_df9['daily returns'].plot()
result_df10['daily returns'].plot()
#%%
result_df1['prices'].plot()
result_df2['prices'].plot()
result_df3['prices'].plot()
result_df4['prices'].plot()
result_df5['prices'].plot()
result_df6['prices'].plot()
result_df7['prices'].plot()
result_df8['prices'].plot()
result_df9['prices'].plot()
result_df10['prices'].plot()

#%%
'''Plot of time series of prices of the different funds in the asset universe'''

#making subplots
figure, axis = plt.subplots(3, 2)
ax=plt.gca()
axis[0, 0].plot(result_df1['prices'],color='k')
axis[0, 0].set_title("Access Sverige A")
axis[0,0].grid()

#No values on x-axis
axis[0,0].get_xaxis().set_visible(False)

axis[0, 1].plot(result_df2['prices'],color='k')
axis[0, 1].set_title("Bas 75")
axis[0,1].grid()
axis[0,1].get_xaxis().set_visible(False)
 
axis[1, 0].plot(result_df3['prices'],color='k')
axis[1, 0].set_title("Allemansfond Komplett")
axis[1,0].grid()
axis[1,0].get_xaxis().set_visible(False)

axis[1, 1].plot(result_df4['prices'],color='k')
axis[1, 1].set_title("USA A")
axis[1,1].grid()
axis[1,1].get_xaxis().set_visible(False)
  
axis[2, 0].plot(result_df5['prices'],color='k')
axis[2, 0].set_title("Europafond A")
axis[2,0].grid()
axis[2,0].get_xaxis().set_visible(False)

axis[2, 1].plot(result_df6['prices'],color='k')
axis[2, 1].set_title("Dynamic A")
axis[2,1].grid()
axis[2,1].get_xaxis().set_visible(False)

plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.4,hspace=0.8)
plt.ylabel('Prices')
plt.show()

figure, axis = plt.subplots(2, 2)

axis[0, 0].plot(result_df7['prices'],color='k')
axis[0, 0].set_title("Bas 50")
axis[0,0].grid()
axis[0,0].get_xaxis().set_visible(False)

axis[0, 1].plot(result_df8['prices'],color='k')
axis[0, 1].set_title("Access Mix A")
axis[0,1].grid()
axis[0,1].get_xaxis().set_visible(False)

axis[1, 0].plot(result_df9['prices'],color='k')
axis[1, 0].set_title("Asienfond A")
axis[1,0].grid()
axis[1,0].get_xaxis().set_visible(False)

axis[1, 1].plot(result_df10['prices'],color='k')
axis[1, 1].set_title("Bas 25")
axis[1,1].grid()
axis[1,1].get_xaxis().set_visible(False)

# Combine all the operations and display
plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.4,hspace=0.5)
plt.show()

