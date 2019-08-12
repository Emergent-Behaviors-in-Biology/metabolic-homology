import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
from scipy.spatial.distance import *

def r2_bc(y_true,y_pred,y_null):
    y_true = np.hstack([y_true,(1-y_true.sum(axis=1))[:,np.newaxis]])
    y_pred = np.hstack([y_pred,(1-y_pred.sum(axis=1))[:,np.newaxis]])
    y_null = np.hstack([y_null,(1-y_null.sum(axis=1))[:,np.newaxis]])
    
    d = []
    for k in range(len(y_true)):
	    d.append(cdist(y_true[k,:][np.newaxis,:],y_pred[k,:][np.newaxis,:],metric='braycurtis'))
    d = np.asarray(d)
    d_null = cdist(y_true,y_null,metric='braycurtis')
    return (d_null-d).mean()/d_null.mean()

def best_alpha_UCB(cv_results):
	UCB = cv_results['mean_test_score']+cv_results['std_test_score']
	return np.argmax(UCB)



class mh_predict:
	def __init__(self,carbon,community,p_carb=10,p_com=10,level='Family',n_train=10,n_test=10,alpha_lasso=1,alpha_ridge=1e4,reduce_dimension=True,test_data=None,norm=None,perform='Bray-Curtis'):

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

		self.perform = perform
		self.train = train
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
		self.alpha_lasso = alpha_lasso
		self.alpha_ridge = alpha_ridge
		self.p_carb = p_carb
		self.p_com = p_com
		self.level = level

		self.X_test = self.X_test.join(carbon_metadata['Category']).set_index('Category',append=True).reorder_levels([3,0,1,2]).astype(float)
		self.Y_test = self.Y_test.join(carbon_metadata['Category']).set_index('Category',append=True).reorder_levels([3,0,1,2]).astype(float)
		self.X_train = self.X_train.join(carbon_metadata['Category']).set_index('Category',append=True).reorder_levels([3,0,1,2]).astype(float)
		self.Y_train = self.Y_train.join(carbon_metadata['Category']).set_index('Category',append=True).reorder_levels([3,0,1,2]).astype(float)
		self.Y_null = self.Y_train.mean().values[np.newaxis,:]

	def run_lasso(self,cross_validate=False,plot=False,lb=-3,ub=2,ns=15,n_splits=50):
		if cross_validate:
			params = {'alpha':np.logspace(lb, ub, ns)}
			self.lasso = GridSearchCV(Lasso(max_iter=10000),params,cv=GroupShuffleSplit(n_splits=n_splits).split(self.X_train,groups=self.X_train.reset_index()['Carbon_Source']),
				refit = best_alpha_UCB)
			self.lasso.fit(self.X_train, self.Y_train)
			self.lasso.coef_ = self.lasso.best_estimator_.coef_
			self.alpha_lasso = self.lasso.best_estimator_.alpha
			if plot:
				for splitname in ['split'+str(k)+'_test_score' for k in range(n_splits)]:
					plt.semilogx(self.lasso.cv_results_['param_alpha'],self.lasso.cv_results_[splitname])
				plt.ylabel(r'Performance')
				plt.xlabel(r'$\alpha$')
				plt.title('Lasso Performance')
				plt.ylim([-0.01, 1.0])
				plt.show()
		else:
			self.lasso =  Lasso(alpha=self.alpha_lasso, max_iter=10000)
			self.lasso.fit(self.X_train, self.Y_train)

		self.Y_pred_lasso = pd.DataFrame(self.lasso.predict(self.X_test),index=self.Y_test.index,columns=self.Y_test.keys())
		if self.perform == 'Bray-Curtis':
			self.r2_train_lasso = r2_bc(self.Y_train.values,self.lasso.predict(self.X_train),self.Y_null)
			self.r2_test_lasso = r2_bc(self.Y_test.values,self.lasso.predict(self.X_test),self.Y_null)
		else:
			self.r2_train_lasso = r2_score(self.Y_train.values,self.lasso.predict(self.X_train),multioutput='variance_weighted')
			self.r2_test_lasso = r2_score(self.Y_test.values,self.lasso.predict(self.X_test),multioutput='variance_weighted')

		return self.lasso.coef_, self.r2_train_lasso, self.r2_test_lasso

	def run_ridge(self,cross_validate=False,plot=False,lb=1,ub=6,ns=15,n_splits=50):
		if cross_validate:
			params = {'alpha':np.logspace(lb, ub, ns)}
			self.ridge = GridSearchCV(Ridge(max_iter=10000),params,cv=GroupShuffleSplit(n_splits=n_splits).split(self.X_train,groups=self.X_train.reset_index()['Carbon_Source']),
				refit = best_alpha_UCB)
			self.ridge.fit(self.X_train, self.Y_train)
			self.ridge.coef_ = self.ridge.best_estimator_.coef_
			self.alpha_ridge = self.ridge.best_estimator_.alpha
			if plot:
				for splitname in ['split'+str(k)+'_test_score' for k in range(n_splits)]:
					plt.semilogx(self.ridge.cv_results_['param_alpha'],self.ridge.cv_results_[splitname])
				plt.ylabel(r'Performance')
				plt.xlabel(r'$\alpha$')
				plt.title('Ridge Performance')
				plt.ylim([-0.01, 1.0])
				plt.show()
		else:
			self.ridge =  Ridge(alpha=self.alpha_ridge, max_iter=10000)
			self.ridge.fit(self.X_train, self.Y_train)

		self.Y_pred_ridge = pd.DataFrame(self.ridge.predict(self.X_test),index=self.Y_test.index,columns=self.Y_test.keys())
		if self.perform == 'Bray-Curtis':
			self.r2_train_ridge = r2_bc(self.Y_train.values,self.ridge.predict(self.X_train),self.Y_null)
			self.r2_test_ridge = r2_bc(self.Y_test.values,self.ridge.predict(self.X_test),self.Y_null)
		else:
			self.r2_train_ridge = r2_score(self.Y_train.values,self.ridge.predict(self.X_train),multioutput='variance_weighted')
			self.r2_test_ridge = r2_score(self.Y_test.values,self.ridge.predict(self.X_test),multioutput='variance_weighted')
	
		return self.ridge.coef_, self.r2_train_ridge, self.r2_test_ridge

	def run_linear(self):
		self.linear=LinearRegression()
		self.linear.fit(self.X_train, self.Y_train)
		self.Y_pred_linear = pd.DataFrame(self.linear.predict(self.X_test),index=self.Y_test.index,columns=self.Y_test.keys())
		if self.perform == 'Bray-Curtis':
			self.r2_train_linear = r2_bc(self.Y_train.values,self.linear.predict(self.X_train),self.Y_null)
			self.r2_test_linear = r2_bc(self.Y_test.values,self.linear.predict(self.X_test),self.Y_null)
		else:
			self.r2_train_linear = r2_score(self.Y_train.values,self.linear.predict(self.X_train),multioutput='variance_weighted')
			self.r2_test_linear = r2_score(self.Y_test.values,self.linear.predict(self.X_test),multioutput='variance_weighted')

		return self.linear.coef_, self.r2_train_linear, self.r2_test_linear

	def run_knn(self):
		Y_train = self.Y_train.groupby(level=0).mean()
		X_train = self.X_train.groupby(level=0).mean()

		self.knn = KNeighborsRegressor(n_neighbors=1)
		self.knn.fit(X_train,Y_train)

		self.Y_pred_knn = pd.DataFrame(self.knn.predict(self.X_test),index=self.Y_test.index,columns=self.Y_test.keys())

		if self.perform == 'Bray-Curtis':
			self.r2_test_knn = r2_bc(self.Y_test.values,self.Y_pred_knn.values,self.Y_null)
		else:
			self.r2_test_knn = r2_score(self.Y_test.values,self.Y_pred_knn.values,multioutput='variance_weighted')

		return self.r2_test_knn

	def run_random_forest(self):
		self.forest = RandomForestRegressor()
		self.forest.fit(self.X_train,self.Y_train)
		self.Y_pred_forest = pd.DataFrame(self.forest.predict(self.X_test),index=self.X_test.index,columns=self.Y_test.keys())

		if self.perform == 'Bray-Curtis':
			self.r2_test_forest = r2_bc(self.Y_test.values,self.Y_pred_forest.values,self.Y_null)
		else:
			self.r2_test_forest = r2_score(self.Y_test.values,self.Y_pred_forest.values,multioutput='variance_weighted')

		return self.forest.feature_importances_, self.r2_test_forest





