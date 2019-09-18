import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import r2_score, make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
import seaborn as sns
import pickle
from scipy.spatial.distance import *

def r2_custom(y_true,y_pred,metric='braycurtis',epsilon=1e-10):
    y_true = np.hstack([y_true,(1-y_true.sum(axis=1))[:,np.newaxis]])
    y_pred = np.hstack([y_pred,(1-y_pred.sum(axis=1))[:,np.newaxis]])
    y_null = y_true.mean(axis=0)[np.newaxis,:]
    
    d = np.diag(cdist(y_true,y_pred,metric=metric))
    d_null = cdist(y_true,y_null,metric=metric).squeeze()

    return (d_null-d).mean()/(d_null.mean()+epsilon)

def best_alpha_UCB(cv_results):
	UCB = cv_results['mean_test_score']+cv_results['std_test_score']
	return np.argmax(UCB)

class mh_predict:
	def __init__(self,carbon,community,p_carb=10,p_com=10,level='Family',n_train=10,n_test=10,reduce_dimension=True,test_data=None,norm=None,metric='braycurtis'):

		#Format tables for carbons
		carbon_table = pd.pivot_table(carbon,values='Flux',columns='Reaction',index='Carbon_Source',aggfunc=np.sum,fill_value=0)
		carbon_metadata = pd.pivot_table(carbon,values='Category',index='Carbon_Source',aggfunc='first')
		if norm is not None:
			carbon_table = carbon_table.div(norm,axis=0)
		#Expand flux data to have one row per experiment
		first = True
		for inoc in range(10):
			for rep in range(10):
				X_new = carbon_table.copy()
				X_new['Inoculum'] = inoc
				X_new['Replicate'] = rep
				X_new = X_new.set_index(['Inoculum','Replicate'],append=True)
				if first:
					X = X_new
					first = False
				else:
					X = X.append(X_new)
		if reduce_dimension:
			#reduce dimension of carbon vector using PCA
			PCA_model = PCA(n_components=p_carb).fit(carbon_table)
			carbon_table = pd.DataFrame(PCA_model.transform(carbon_table),columns=['PC '+str(k+1) for k in range(p_carb)],index=carbon_table.index)


		#Format tables for community
		Y = pd.pivot_table(community,values='Relative_Abundance',columns=level,index=['Carbon_Source','Inoculum','Replicate'],aggfunc=np.sum,fill_value=0)
		#Find carbon sources for which no flux data is available
		no_data = list(set(community['Carbon_Source'])-set(carbon_table.index)) 
		if len(no_data) > 0:
			print('Dropped training CS missing from carbon data: '+', '.join(no_data))
		Y = Y.drop(no_data)
		#Keep only top p_com most abundant families, and make sure they are also in test data
		chosen = Y.sum().sort_values(ascending=False).index[:p_com]
		if test_data is not None:
			no_data_test = list(set(test_data['Carbon_Source'])-set(carbon_table.index))
			if len(no_data_test) > 0:
				print('Dropped test CS missing from carbon data: '+', '.join(no_data_test))
			chosen = list(set(chosen).intersection(set(test_data[level])))
		Y = Y.T.loc[chosen].T
    
    	#Create training and test sets with non-overlapping carbon sources, 
    	#ensuring that the training set has at least one sugar and one acid       
		go = True
		k=0
		while go and k<1000:
			if test_data is None:
				train = np.random.choice(list(set(community['Carbon_Source'])-set(no_data)),size=n_train,replace=False)
				test = list(set(community['Carbon_Source'])-set(no_data)-set(train))
			else:
				if len(set(test_data['Carbon_Source'])-set(no_data_test))<len(set(community['Carbon_Source'])):
					test = np.random.choice(list(set(test_data['Carbon_Source'])-set(no_data_test)),size=n_test,replace=False)
					train = list(set(community['Carbon_Source'])-set(no_data)-set(test))
				else:
					train = np.random.choice(list(set(community['Carbon_Source'])-set(no_data)),size=n_train,replace=False)
					test = list(set(test_data['Carbon_Source'])-set(no_data_test)-set(train))
			t = list(carbon_metadata.reindex(train)['Category'])
			k+=1
			if 'F' in t and 'R' in t:
				go = False

		#Save test and train data
		self.Y_train = Y.loc[train]
		self.X_train = X.reindex(self.Y_train.index)
		if test_data is not None:
			self.Y_test = pd.pivot_table(test_data,values='Relative_Abundance',columns=level,index=['Carbon_Source','Inoculum','Replicate'],aggfunc=np.sum,fill_value=0)
			self.Y_test = self.Y_test.drop(no_data_test)
			self.Y_test = self.Y_test.T.loc[chosen].T
			self.Y_test = self.Y_test.loc[test]
		else:
			self.Y_test = Y.loc[test]
		self.X_test = X.reindex(self.Y_test.index)
		self.X_test = self.X_test.join(carbon_metadata['Category']).set_index('Category',append=True).reorder_levels([3,0,1,2]).astype(float)
		self.Y_test = self.Y_test.join(carbon_metadata['Category']).set_index('Category',append=True).reorder_levels([3,0,1,2]).astype(float)
		self.X_train = self.X_train.join(carbon_metadata['Category']).set_index('Category',append=True).reorder_levels([3,0,1,2]).astype(float)
		self.Y_train = self.Y_train.join(carbon_metadata['Category']).set_index('Category',append=True).reorder_levels([3,0,1,2]).astype(float)
		self.Y_null = self.Y_train.mean().values[np.newaxis,:]

		#Set default values and initialize variables
		self.metric = metric
		self.scorer = make_scorer(r2_custom,metric=self.metric)
		self.train = train
		self.alpha_lasso = 1e-2
		self.alpha_ridge = 1e2
		self.alpha_net = 1e-2
		self.alpha_kridge = 1e-2
		self.gamma_kridge = 1
		self.l1_ratio_net = 0.1
		self.p_carb = p_carb
		self.p_com = p_com
		self.level = level
		self.train_score = {}
		self.test_score = {}
		self.Y_pred = {}
		self.estimators = {}

	def run_lasso(self,cross_validate=False,plot=False,lb=-3,ub=2,ns=15,n_splits=50):
		if cross_validate:
			params = {'alpha':np.logspace(lb, ub, ns)}
			self.lasso = GridSearchCV(Lasso(max_iter=100000),params,cv=GroupShuffleSplit(n_splits=n_splits).split(self.X_train,groups=self.X_train.reset_index()['Carbon_Source']),
				refit = best_alpha_UCB, scoring = self.scorer)
			self.lasso.fit(self.X_train, self.Y_train)
			self.lasso.coef_ = self.lasso.best_estimator_.coef_
			self.alpha_lasso = self.lasso.best_estimator_.alpha
			self.estimators['LASSO'] = self.lasso.best_estimator_
			if plot:
				for splitname in ['split'+str(k)+'_test_score' for k in range(n_splits)]:
					plt.semilogx(self.lasso.cv_results_['param_alpha'],self.lasso.cv_results_[splitname])
				plt.ylabel('Performance ('+self.metric+')')
				plt.xlabel(r'$\alpha$')
				plt.title('Lasso Performance')
				plt.ylim([-0.01, 1.0])
				plt.show()
		else:
			self.lasso =  Lasso(alpha=self.alpha_lasso, max_iter=100000)
			self.lasso.fit(self.X_train, self.Y_train)
			self.estimators['LASSO'] = self.lasso

		self.Y_pred['LASSO'] = pd.DataFrame(self.lasso.predict(self.X_test),index=self.Y_test.index,columns=self.Y_test.keys())
		self.train_score['LASSO'] = r2_custom(self.Y_train.values,self.lasso.predict(self.X_train),metric=self.metric)
		self.test_score['LASSO'] = r2_custom(self.Y_test.values,self.lasso.predict(self.X_test),metric=self.metric)

	def run_ridge(self,cross_validate=False,plot=False,lb=1,ub=6,ns=15,n_splits=20):
		if cross_validate:
			params = {'alpha':np.logspace(lb, ub, ns)}
			self.ridge = GridSearchCV(Ridge(max_iter=10000),params,cv=GroupShuffleSplit(n_splits=n_splits).split(self.X_train,groups=self.X_train.reset_index()['Carbon_Source']),
				refit = best_alpha_UCB, scoring = self.scorer)
			self.ridge.fit(self.X_train, self.Y_train)
			self.ridge.coef_ = self.ridge.best_estimator_.coef_
			self.alpha_ridge = self.ridge.best_estimator_.alpha
			self.estimators['Ridge'] = self.ridge.best_estimator_
			if plot:
				for splitname in ['split'+str(k)+'_test_score' for k in range(n_splits)]:
					plt.semilogx(self.ridge.cv_results_['param_alpha'],self.ridge.cv_results_[splitname])
				plt.ylabel('Performance ('+self.metric+')')
				plt.xlabel(r'$\alpha$')
				plt.title('Ridge Performance')
				plt.ylim([-0.01, 1.0])
				plt.show()
		else:
			self.ridge =  Ridge(alpha=self.alpha_ridge, max_iter=10000)
			self.ridge.fit(self.X_train, self.Y_train)
			self.estimators['Ridge'] = self.ridge

		self.Y_pred['Ridge'] = pd.DataFrame(self.ridge.predict(self.X_test),index=self.Y_test.index,columns=self.Y_test.keys())
		self.train_score['Ridge'] = r2_custom(self.Y_train.values,self.ridge.predict(self.X_train),metric=self.metric)
		self.test_score['Ridge'] = r2_custom(self.Y_test.values,self.ridge.predict(self.X_test),metric=self.metric)

	def run_elastic_net(self,cross_validate=False,plot=False,lb=[-3,-2],ub=[2,0],ns=5,n_splits=20):
		if cross_validate:
			params = {'alpha':np.logspace(lb[0], ub[0], ns), 'l1_ratio':np.logspace(lb[1], ub[1], ns)}
			self.net = GridSearchCV(ElasticNet(max_iter=10000),params,cv=GroupShuffleSplit(n_splits=n_splits).split(self.X_train,groups=self.X_train.reset_index()['Carbon_Source']),
				refit = best_alpha_UCB, scoring = self.scorer)
			self.net.fit(self.X_train, self.Y_train)
			self.net.coef_ = self.net.best_estimator_.coef_
			self.alpha_net = self.net.best_estimator_.alpha
			self.l1_ratio_net = self.net.best_estimator_.l1_ratio
			self.estimators['Elastic Net'] = self.net.best_estimator_
			if plot:
				sns.heatmap(pd.pivot_table(pd.DataFrame(self.net.cv_results_),index='param_alpha',columns='param_l1_ratio',values='mean_test_score') +
					pd.pivot_table(pd.DataFrame(self.net.cv_results_),index='param_alpha',columns='param_l1_ratio',values='std_test_score'))
				plt.title('Elastic Net Performance')
				plt.show()
		else:
			self.net =  ElasticNet(alpha=self.alpha_net,l1_ratio=self.l1_ratio_net, max_iter=10000)
			self.net.fit(self.X_train, self.Y_train)
			self.estimators['Elastic Net'] = self.net

		self.Y_pred['Elastic Net'] = pd.DataFrame(self.net.predict(self.X_test),index=self.Y_test.index,columns=self.Y_test.keys())
		self.train_score['Elastic Net'] = r2_custom(self.Y_train.values,self.net.predict(self.X_train),metric=self.metric)
		self.test_score['Elastic Net'] = r2_custom(self.Y_test.values,self.net.predict(self.X_test),metric=self.metric)

	def run_kernel_ridge(self,cross_validate=False,plot=False,lb=[-2,-3],ub=[2,0],ns=5,n_splits=20,kernel='rbf'):
		if cross_validate:
			params = {'alpha':np.logspace(lb[0], ub[0], ns), 'gamma':np.logspace(lb[1], ub[1], ns)}
			self.kridge = GridSearchCV(KernelRidge(kernel=kernel),params,cv=GroupShuffleSplit(n_splits=n_splits).split(self.X_train,groups=self.X_train.reset_index()['Carbon_Source']),
				refit = best_alpha_UCB, scoring = self.scorer)
			self.kridge.fit(self.X_train, self.Y_train)
			self.alpha_kridge = self.kridge.best_estimator_.alpha
			self.gamma_kridge = self.kridge.best_estimator_.gamma
			self.estimators['Kernel Ridge'] = self.kridge.best_estimator_
			if plot:
				sns.heatmap(pd.pivot_table(pd.DataFrame(self.kridge.cv_results_),index='param_alpha',columns='param_gamma',values='mean_test_score') +
					pd.pivot_table(pd.DataFrame(self.kridge.cv_results_),index='param_alpha',columns='param_gamma',values='std_test_score'))
				plt.title('Kernel Ridge Performance')
				plt.show()
		else:
			self.kridge = KernelRidge(alpha=self.alpha_kridge,gamma=self.gamma_kridge)
			self.kridge.fit(self.X_train, self.Y_train)
			self.estimators['Kernel Ridge'] = self.kridge

		self.Y_pred['Kernel Ridge'] = pd.DataFrame(self.kridge.predict(self.X_test),index=self.Y_test.index,columns=self.Y_test.keys())
		self.train_score['Kernel Ridge'] = r2_custom(self.Y_train.values,self.kridge.predict(self.X_train),metric=self.metric)
		self.test_score['Kernel Ridge'] = r2_custom(self.Y_test.values,self.kridge.predict(self.X_test),metric=self.metric)

	def run_linear(self):
		self.linear=LinearRegression()
		self.linear.fit(self.X_train, self.Y_train)

		self.Y_pred['OLS'] = pd.DataFrame(self.linear.predict(self.X_test),index=self.Y_test.index,columns=self.Y_test.keys())
		self.train_score['OLS'] = r2_custom(self.Y_train.values,self.linear.predict(self.X_train),metric=self.metric)
		self.test_score['OLS'] = r2_custom(self.Y_test.values,self.linear.predict(self.X_test),metric=self.metric)
		self.estimators['OLS'] = self.linear

	def run_knn(self):
		Y_train = self.Y_train.groupby(level=0).mean()
		X_train = self.X_train.groupby(level=0).mean()

		self.knn = KNeighborsRegressor(n_neighbors=1)
		self.knn.fit(X_train,Y_train)

		self.Y_pred['KNN'] = pd.DataFrame(self.knn.predict(self.X_test),index=self.Y_test.index,columns=self.Y_test.keys())
		self.train_score['KNN'] = r2_custom(self.Y_train.values,self.knn.predict(self.X_train),metric=self.metric)
		self.test_score['KNN'] = r2_custom(self.Y_test.values,self.knn.predict(self.X_test),metric=self.metric)
		self.estimators['KNN'] = self.knn

	def run_random_forest(self):
		self.forest = RandomForestRegressor(n_estimators=100)
		self.forest.fit(self.X_train,self.Y_train)
		self.Y_pred['Random Forest'] = pd.DataFrame(self.forest.predict(self.X_test),index=self.X_test.index,columns=self.Y_test.keys())

		self.train_score['Random Forest'] = r2_custom(self.Y_train.values,self.forest.predict(self.X_train),metric=self.metric)
		self.test_score['Random Forest'] = r2_custom(self.Y_test.values,self.forest.predict(self.X_test),metric=self.metric)
		self.estimators['Random Forest'] = self.forest

	def regenerate_predictions(self):
		for method in self.estimators.keys():
			self.Y_pred[method] = pd.DataFrame(self.estimators[method].predict(self.X_test),index=self.Y_test.index,columns=self.Y_test.keys())
			self.train_score[method] = r2_custom(self.Y_train.values,self.estimators[method].predict(self.X_train),metric=self.metric)
			self.test_score[method] = r2_custom(self.Y_test.values,self.estimators[method].predict(self.X_test),metric=self.metric)



