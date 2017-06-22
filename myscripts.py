import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import cross_val_score
def create_stats(df,window):
	#if you get infs, it's because your data was 0
	
	for each_col_name in df.columns:
		df[each_col_name].replace(to_replace=0, value=0.001, inplace=True)
		df[each_col_name+"RollingMean"]=df[each_col_name].rolling(window=window).mean()
		#df[each_col_name+"RollingStd"]=df[each_col_name].rolling(window=window).std()
		df[each_col_name+"RelAvg"]=(df[each_col_name]/df[each_col_name+"RollingMean"])-1
		#df[each_col_name+"Std"]=(df[each_col_name]-df[each_col_name+"RollingMean"])/df[each_col_name+"RollingStd"]
		df=df.drop(each_col_name,axis=1)
	df=df.ix[window-1:];
	df.fillna(0);
	return df
	
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
	
	#Determing rolling statistics
	rolmean =timeseries.rolling(window=12,center=False).mean()
	rolstd = timeseries.rolling(window=12,center=False).std()
	#rolmean = pd.rolling_mean(timeseries, window=12)
	#rolstd = pd.rolling_std(timeseries, window=12)

	#Plot rolling statistics:
	orig = plt.plot(timeseries, color='blue',label='Original')
	mean = plt.plot(rolmean, color='red', label='Rolling Mean')
	std = plt.plot(rolstd, color='black', label = 'Rolling Std')
	plt.legend(loc='best')
	plt.title('Rolling Mean & Standard Deviation')
	plt.show(block=False)
	
	#Perform Dickey-Fuller test:
	print ('Results of Dickey-Fuller Test:')
	dftest = adfuller(timeseries, autolag='AIC')
	dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	for key,value in dftest[4].items():
		dfoutput['Critical Value (%s)'%key] = value
	print (dfoutput)
	
def rmse_cv(model):
	rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
	#with scoring being blank, by default this would've outputted the accuracy, ex: 95%
	#with scoring="neg_mean_squared_error", we get accuracy -1, so shows by how much you were off and it's negative
	#then with the - in front, gives you the error, but positive. 
	return(rmse)
	
	
	
def prep_hockey_data(hockey_data):

	hockey_data.columns=['Away','Home']
	hockey_data["MtlGoals"]=hockey_data.Home
	hockey_data["OppGoals"]=hockey_data.Home

	#separating the goals from the name
	away_goals=hockey_data.Away.str.extract('(\d+)').astype(int)
	home_goals=hockey_data.Home.str.extract('(\d+)').astype(int)

	#this version will cause the 'A value is trying to be set on a copy of a slice from a DataFrame.' problem
	#hockey_data[hockey_data.Away.str.contains("MTL")]."MtlGoals"=away_goals[hockey_data.Away.str.contains("MTL")].values

	mtl_home=hockey_data.Home.str.contains("MTL")
	mtl_away=~hockey_data.Home.str.contains("MTL")

	hockey_data.loc[mtl_away,"MtlGoals"]=away_goals[mtl_away].values
	hockey_data.loc[mtl_home,"MtlGoals"]=home_goals[mtl_home].values

	hockey_data.loc[mtl_home,"OppGoals"]=away_goals[mtl_home].values
	hockey_data.loc[mtl_away,"OppGoals"]=home_goals[mtl_away].values

	hockey_data.Away=hockey_data.Away.str.replace('\d+', '')
	hockey_data.Home=hockey_data.Home.str.replace('\d+', '')

	hockey_data["Opp"]=hockey_data.Away
	hockey_data.loc[mtl_away,"Opp"]=hockey_data.Home[mtl_away].values
	hockey_data.loc[mtl_home,"Opp"]=hockey_data.Away[mtl_home].values

	#I need these to be numbers because later on I will be summing them up

	hockey_data.loc[mtl_away,"Away"]=1
	hockey_data.loc[mtl_away,"Home"]=0

	hockey_data.loc[mtl_home,"Away"]=0
	hockey_data.loc[mtl_home,"Home"]=1

	hockey_data["Win"]=0
	hockey_data["Tie"]=0
	hockey_data["Defeat"]=0


	wins=hockey_data.MtlGoals>hockey_data.OppGoals
	ties=hockey_data.MtlGoals==hockey_data.OppGoals
	losses=hockey_data.MtlGoals<hockey_data.OppGoals

	hockey_data.loc[wins,"Win"]=1
	hockey_data.loc[ties,"Tie"]=1
	hockey_data.loc[losses,"Defeat"]=1

	#days of the week

	hockey_data["monday"]=hockey_data.index.dayofweek==0
	hockey_data["tuesday"]=hockey_data.index.dayofweek==1
	hockey_data["wednesday"]=hockey_data.index.dayofweek==2
	hockey_data["thursday"]=hockey_data.index.dayofweek==3
	hockey_data["friday"]=hockey_data.index.dayofweek==4
	hockey_data["saturday"]=hockey_data.index.dayofweek==5
	hockey_data["sunday"]=hockey_data.index.dayofweek==6

	hockey_data.Away=pd.to_numeric(hockey_data.Away);
	hockey_data.Home=pd.to_numeric(hockey_data.Home);
	hockey_data.MtlGoals=pd.to_numeric(hockey_data.MtlGoals);
	hockey_data.OppGoals=pd.to_numeric(hockey_data.OppGoals);

	hockey_data=hockey_data.sort_index()

	monthly_hockey_data=hockey_data.resample("M").sum();
	monthly_hockey_data=monthly_hockey_data.dropna()
	return monthly_hockey_data


def normalize_df(df):
	#df=pd.DataFrame(df)
	#
	df["short_term_mean"]=df.number_of_times.rolling(6).mean()
	where_outliers_be=np.abs(df.number_of_times-df.number_of_times.mean())>=(3*df.number_of_times.std())
	
	df.number_of_times[where_outliers_be]=df.short_term_mean[where_outliers_be]
	
	normalized_df= (df - df.mean()) / (df.max() - df.min())
	normalized_df=pd.drop(normalized_df.short_term_mean,1)
	return normalized_df
