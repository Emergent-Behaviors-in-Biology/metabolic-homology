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
	def __init__(self,carbon,community,p_carb=10,p_com=10,level='Family',n_train=10,alpha_lasso=1,alpha_ridge=1e4):

		#Reduce dimension of flux data
		carbon_table = pd.pivot_table(carbon,values='Flux',columns='Reaction',index='Carbon_Source',aggfunc=np.sum,fill_value=0)
		carbon_metadata = pd.pivot_table(carbon,values='Type',index='Carbon_Source',aggfunc='first')
		PCA_model = PCA(n_components=p_carb).fit(carbon_table)
		explained_variance = np.around(100*PCA_model.explained_variance_ratio_,decimals=1)
		carbon_table = pd.DataFrame(PCA_model.transform(carbon_table),columns=['PC '+str(k+1) for k in range(p_carb)],index=carbon_table.index)

		#Find carbon sources for which no flux data is available
		no_data = list(set(community['Carbon_Source'])-set(carbon_table.index)) 

		#Reduce dimension of community data
		Y = pd.pivot_table(community,values='Relative_Abundance',columns=level,index=['Carbon_Source','Inoculum','Replicate'],aggfunc=np.sum,fill_value=0)
		Y = Y.drop(no_data)
		chosen = Y.sum().sort_values(ascending=False).index[:p_com]
		Y = Y.T.loc[chosen].T

		#Expand flux data to have one row per experiment
		first = True
		for inoc in Y.index.levels[1]:
			for rep in Y.index.levels[2]:
				X_new = carbon_table.copy()
				X_new['Inoculum'] = inoc
				X_new['Replicate'] = rep
				X_new = X_new.set_index(['Inoculum','Replicate'],append=True)
				if first:
					X = X_new
					first = False
				else:
					X = X.append(X_new)
    
    	#Create training and test sets, ensuring that the training set has at least one sugar and one acid       
		go = True
		k=0
		while go and k<1000:
			train = np.random.choice(list(set(Y.index.levels[0])-set(no_data)),size=n_train,replace=False)
			test = list(set(Y.index.levels[0])-set(no_data)-set(train))
			t = list(carbon_metadata.reindex(train)['Type'])
			k+=1
			if 'A' in t and ('MS' in t or 'DS' in t):
				go = False
		del carbon_metadata

		self.Y_train = Y.loc[train]
		self.X_train = X.reindex(self.Y_train.index)
		self.Y_test = Y.loc[test]
		self.X_test = X.reindex(self.Y_test.index)
		self.alpha_lasso = alpha_lasso
		self.alpha_ridge = alpha_ridge
		self.p_carb = p_carb
		self.p_com = p_com
		self.level = level

	def run_lasso(self,cross_validate=False,plot=False):
		self.lasso=Lasso()

		if cross_validate:
			alphas = np.logspace(-2, 2, 10)
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

	def run_ridge(self,cross_validate=False,plot=False):
		self.ridge=Ridge()

		if cross_validate:
			alphas = np.logspace(2, 6, 10)
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




