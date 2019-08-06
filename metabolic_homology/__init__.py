import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
import pickle
from scipy.spatial.distance import *

class mh_predict:
	def __init__(self,carbon,community,p_carb=10,p_com=10,level='Family',n_train=10,n_test=10,alpha_lasso=1,alpha_ridge=1e4,reduce_dimension=True,test_data=None):

		#Format tables for carbons
		carbon_table = pd.pivot_table(carbon,values='Flux',columns='Reaction',index='Carbon_Source',aggfunc=np.sum,fill_value=0)
		carbon_metadata = pd.pivot_table(carbon,values='Category',index='Carbon_Source',aggfunc='first')
		#Find carbon sources for which no flux data is available
		no_data = list(set(community['Carbon_Source'])-set(carbon_table.index)) 
		if len(no_data) > 0:
			print('Dropped training CS missing from carbon data: '+', '.join(no_data))
		#Format tables for community
		Y = pd.pivot_table(community,values='Relative_Abundance',columns=level,index=['Carbon_Source','Inoculum','Replicate'],aggfunc=np.sum,fill_value=0)
		Y = Y.drop(no_data)
		#Keep only top p_com most abundant families
		chosen = Y.sum().sort_values(ascending=False).index[:p_com]
		if test_data is not None:
			no_data_test = list(set(test_data['Carbon_Source'])-set(carbon_table.index))
			if len(no_data_test) > 0:
				print('Dropped test CS missing from carbon data: '+', '.join(no_data_test))
			chosen = list(set(chosen).intersection(set(test_data[level])))
		Y = Y.T.loc[chosen].T

		if reduce_dimension:
			#reduce dimension of carbon vector using PCA
			PCA_model = PCA(n_components=p_carb).fit(carbon_table)
			carbon_table = pd.DataFrame(PCA_model.transform(carbon_table),columns=['PC '+str(k+1) for k in range(p_carb)],index=carbon_table.index)

		#Expand flux data to have one row per experiment
		first = True
		for inoc in set(community['Inoculum']):
			for rep in set(community['Replicate']):
				X_new = carbon_table.copy()
				X_new['Inoculum'] = inoc
				X_new['Replicate'] = rep
				X_new = X_new.set_index(['Inoculum','Replicate'],append=True)
				if first:
					X = X_new
					first = False
				else:
					X = X.append(X_new)
		if test_data is not None:
			first = True
			for inoc in set(test_data['Inoculum']):
				for rep in set(test_data['Replicate']):
					X_new = carbon_table.copy()
					X_new['Inoculum'] = inoc
					X_new['Replicate'] = rep
					X_new = X_new.set_index(['Inoculum','Replicate'],append=True)
					if first:
						self.X_test = X_new
						first = False
					else:
						self.X_test = self.X_test.append(X_new)
    
    	#Create training and test sets, ensuring that the training set has at least one sugar and one acid       
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

		self.train = train
		self.Y_train = Y.loc[train]
		self.X_train = X.reindex(self.Y_train.index)
		if test_data is not None:
			self.Y_test = pd.pivot_table(test_data,values='Relative_Abundance',columns=level,index=['Carbon_Source','Inoculum','Replicate'],aggfunc=np.sum,fill_value=0)
			self.Y_test = self.Y_test.drop(no_data_test)
			self.Y_test = self.Y_test.T.loc[chosen].T
			self.Y_test = self.Y_test.loc[test]
			self.X_test = self.X_test.reindex(self.Y_test.index)
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

	def run_lasso(self,cross_validate=False,plot=False,lb=-3,ub=2,ns=15):
		self.lasso=Lasso(max_iter=10000)

		if cross_validate:
			alphas = np.logspace(lb, ub, ns)
			r2_train = []
			r2_test = []
			coeffs = []
			for a in alphas:
				self.lasso.set_params(alpha=a)
				self.lasso.fit(self.X_train, self.Y_train)
				r2_train.append(r2_score(self.Y_train.values,self.lasso.predict(self.X_train),multioutput='variance_weighted'))
				r2_test.append(r2_score(self.Y_test.values,self.lasso.predict(self.X_test),multioutput='variance_weighted'))
				coeffs.append(self.lasso.coef_.reshape(-1))
			self.alpha_lasso = alphas[np.argmax(r2_test)]
			if plot:
				plt.semilogx(alphas,np.abs(coeffs))
				plt.xlabel(r'$\alpha$')
				plt.ylabel(r'$|w_i|$')
				plt.title('Lasso Coefficients')
				plt.show()
				plt.semilogx(alphas,r2_train,'k',label='Train')
				plt.semilogx(alphas,r2_test,'k--',label='Test')
				plt.ylabel(r'Performance ($R^2$)')
				plt.xlabel(r'$\alpha$')
				plt.title('Lasso Performance')
				plt.ylim([-0.01, 1.0])
				plt.legend()
				plt.show()

		self.lasso.set_params(alpha=self.alpha_lasso)
		self.lasso.fit(self.X_train, self.Y_train)
		self.Y_pred_lasso = pd.DataFrame(self.lasso.predict(self.X_test),index=self.Y_test.index,columns=self.Y_test.keys())
		self.r2_train_lasso = r2_score(self.Y_train.values,self.lasso.predict(self.X_train),multioutput='variance_weighted')
		self.r2_test_lasso = r2_score(self.Y_test.values,self.lasso.predict(self.X_test),multioutput='variance_weighted')

		return self.lasso.coef_, self.r2_train_lasso, self.r2_test_lasso

	def run_ridge(self,cross_validate=False,plot=False,lb=1,ub=6,ns=15):
		self.ridge=Ridge(max_iter=10000)

		if cross_validate:
			alphas = np.logspace(lb, ub, ns)
			r2_train = []
			r2_test = []
			coeffs = []
			for a in alphas:
				self.ridge.set_params(alpha=a)
				self.ridge.fit(self.X_train, self.Y_train)
				r2_train.append(r2_score(self.Y_train.values,self.ridge.predict(self.X_train),multioutput='variance_weighted'))
				r2_test.append(r2_score(self.Y_test.values,self.ridge.predict(self.X_test),multioutput='variance_weighted'))
				coeffs.append(self.ridge.coef_.reshape(-1))
			self.alpha_ridge = alphas[np.argmax(r2_test)]
			if plot:
				plt.semilogx(alphas,np.abs(coeffs))
				plt.xlabel(r'$\alpha$')
				plt.ylabel(r'$|w_i|$')
				plt.title('Ridge Coefficients')
				plt.show()
				plt.semilogx(alphas,r2_train,'k',label='Train')
				plt.semilogx(alphas,r2_test,'k--',label='Test')
				plt.ylabel(r'Performance ($R^2$)')
				plt.xlabel(r'$\alpha$')
				plt.title('Ridge Performance')
				plt.ylim([-0.01, 1.0])
				plt.legend()
				plt.show()

		self.ridge.set_params(alpha=self.alpha_ridge)
		self.ridge.fit(self.X_train, self.Y_train)
		self.Y_pred_ridge = pd.DataFrame(self.ridge.predict(self.X_test),index=self.Y_test.index,columns=self.Y_test.keys())
		self.r2_train_ridge = r2_score(self.Y_train.values,self.ridge.predict(self.X_train),multioutput='variance_weighted')
		self.r2_test_ridge = r2_score(self.Y_test.values,self.ridge.predict(self.X_test),multioutput='variance_weighted')

		return self.ridge.coef_, self.r2_train_ridge, self.r2_test_ridge

	def run_linear(self):
		self.linear=LinearRegression()
		self.linear.fit(self.X_train, self.Y_train)
		self.Y_pred_linear = pd.DataFrame(self.linear.predict(self.X_test),index=self.Y_test.index,columns=self.Y_test.keys())
		self.r2_train_linear = r2_score(self.Y_train.values,self.linear.predict(self.X_train),multioutput='variance_weighted')
		self.r2_test_linear = r2_score(self.Y_test.values,self.linear.predict(self.X_test),multioutput='variance_weighted')

		return self.linear.coef_, self.r2_train_linear, self.r2_test_linear

	def run_knn(self,metric='euclidean'):
		Y_train = self.Y_train.groupby(level=0).mean()
		X_train = self.X_train.groupby(level=0).mean()
		Y_pred = self.Y_test.copy()

		for idx in self.X_test.index:
			x = np.ones((len(X_train),1)).dot(self.X_test.loc[idx].values[np.newaxis,:])
			nn = X_train.index[np.argmin(cdist(x,X_train.values,metric = metric))]
			Y_pred.loc[idx] = Y_train.loc[nn]
		self.r2_test_KNN = r2_score(self.Y_test.values,Y_pred.values,multioutput='variance_weighted')
		self.Y_pred_KNN = Y_pred

		return self.r2_test_KNN




