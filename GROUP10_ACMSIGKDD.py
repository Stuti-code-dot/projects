#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
train_file = sys.argv[1]
label_file = sys.argv[2]


# In[ ]:


# import libraries
import pandas as pd
import numpy as np


# In[ ]:


# reading train data
train_x = pd.read_csv(train_file)
train_x


# In[ ]:


train_x = train_x.replace('?',0)
train_x


# In[ ]:


y = pd.read_csv(label_file)
train_y = y.iloc[:,0]
train_y = train_y.replace(-1,0) # benign
train_y


# In[ ]:


ul, c = np.unique(train_y, return_counts = True)
ul,c


# In[ ]:


import seaborn as sns
ax = sns.barplot(x = ul.astype(int), y = c)
ax.set(xlabel='Labels',ylabel = 'Count',title = 'Label Distribution ')


# In[ ]:


from imblearn.under_sampling import NearMiss
ak = NearMiss(sampling_strategy = 'majority',n_jobs = -1)
train_x_rs, train_y_rs = ak.fit_resample(train_x, train_y)


# In[ ]:


train_x_rs.shape


# In[ ]:


np.unique(train_x_rs, return_counts = True)


# In[ ]:


# Applying the genetic algorithm

def genetic(data,labels,i):
    if i==1:
        from sklearn.neural_network import MLPClassifier
        estimator = MLPClassifier(hidden_layer_sizes = (1),activation = 'identity',validation_fraction = 0.1,solver = 'adam',n_iter_no_change = 2,tol = 1e-2,batch_size = 4,learning_rate='adaptive',random_state=1,early_stopping = True)
        from genetic_selection import GeneticSelectionCV
        selector = GeneticSelectionCV(estimator,
                                          cv= 10,
                                          verbose=1,
                                          scoring= 'balanced_accuracy',                                 
                                          max_features=50,
                                          n_population=1246,
                                          crossover_proba=0.5,
                                          mutation_proba=0.2,
                                          n_generations=0,
                                          crossover_independent_proba=0.5,
                                          mutation_independent_proba=0.05,
                                          tournament_size=1,
                                          n_gen_no_change=100,
                                          caching=True,
                                          n_jobs=1)

        selector.fit(data,labels)
        val = selector.support_
        # selecting features
        lst = []
        for i in range(len(val)):
            if (val[i] == False):
                lst.append(i)
        return lst
# all features taken
    elif i==2:
        lst = []
        return lst      


# In[ ]:


final_list = genetic(train_x_rs,train_y_rs,2)
final_list


# In[ ]:


train_x_rs = np.delete(train_x_rs.values,final_list,axis = 1)


# In[ ]:


# run


# In[ ]:


train_x_rs.shape


# In[ ]:


ul1, c1 = np.unique(train_y_rs, return_counts = True)
ul1,c1


# In[ ]:


import seaborn as sns
ax = sns.barplot(x = ul1.astype(int), y = c1)
ax.set(xlabel='Labels',ylabel = 'Count',title = 'Label Distribution')


# In[ ]:


## creating 10 fold train-test spilt

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)


# using different classifiers by defining a function 
###################
def classifier(i):

    if i==1:
        # first classifier (SVM - diff params)
        from sklearn.svm import SVC
        clf = SVC(kernel = 'rbf',random_state=1)
    elif i==2:
        # second classifier (NuSVM)
        from sklearn.svm import NuSVC
        clf = NuSVC(kernel = 'rbf', random_state=1)
    elif i==3:
        # third classifier (Bagging Classifier)
        from sklearn.ensemble import BaggingClassifier
        from sklearn.svm import NuSVC
        clf = BaggingClassifier(base_estimator=NuSVC(kernel = 'rbf', random_state=1),n_estimators=20, bootstrap_features = True,random_state=1,max_samples = 0.9,max_features=3,n_jobs = -1)
    
    elif i==4:
        # fourth classifier (Boosting Classifier)
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.svm import NuSVC
        clf = AdaBoostClassifier(base_estimator=NuSVC(kernel = 'rbf',random_state=1),algorithm = 'SAMME',n_estimators=10,random_state=1)
        
    elif i==5:
        # fifth classifier (Random Forest)
        from sklearn.ensemble import RandomForestClassifier as rf
        clf = rf(n_estimators = 300,oob_score = True,random_state = 1,class_weight = 'balanced',max_features = 'sqrt',n_jobs = -1)
    
    elif i==6:
        # sixth classifier (Gradient Boosting)
        from sklearn.ensemble import GradientBoostingClassifier as gbc
        clf = gbc(n_estimators = 50, subsample = 0.75, max_depth = 50, random_state = 1, learning_rate = 0.5, loss = 'exponential',max_features = 'sqrt')
    
    elif i==7:
        # seventh classifier (XGboost)
        import xgboost as xgb
        clf = xgb.XGBClassifier(booster='gbtree',n_estimators = 50, learning_rate = 0.5,random_state = 1,max_depth=100,subsample = 0.9,n_jobs=-1, num_parallel_tree=15,use_label_encoder=False)
        
    elif i==8:
        # eighth classifier (LightGBM)
        import lightgbm as lgbm
        clf = lgbm.LGBMClassifier(n_estimators = 100,num_leaves = 400,class_weight = 'balanced',random_state = 1, learning_rate = 0.5,max_depth=400,subsample = 0.9,n_jobs=-1)
    
    return(clf)
###################


# calling the function for different classifiers
accuracies =[]
clfreport=[]
rascores=[]
mcc=[]
f1 = []

import time
start_time = time.time()

model = classifier(2)

print("--> Classifier:", model)
i = 1
      
for train_index, test_index in skf.split(train_x_rs, train_y_rs):

    print("--> Fold:", i)
    x_tr,x_te = train_x_rs[train_index], train_x_rs[test_index]
    y_tr,y_te = train_y_rs[train_index], train_y_rs[test_index]
    from sklearn.metrics import accuracy_score
    model.fit(x_tr,y_tr)
    y_pred_val = model.predict(x_te)


    print("--> Accuracy:",accuracy_score(y_te,y_pred_val))
    a1 = accuracy_score(y_te,y_pred_val)
    print("\n")
    from sklearn.metrics import classification_report,roc_auc_score,matthews_corrcoef,f1_score
    print("--> Classification Report: \n",classification_report(y_te,y_pred_val))
    a2 = classification_report(y_te,y_pred_val)
    print("--> ROC_AUC_Score: \n",roc_auc_score(y_te,y_pred_val))
    a3 = roc_auc_score(y_te,y_pred_val)
    print("--> MCC:", matthews_corrcoef(y_te.values.reshape(-1,1),y_pred_val))
    a4 = matthews_corrcoef(y_te.values.reshape(-1,1),y_pred_val)
    print("--> F1 Score:", f1_score(y_te.values.reshape(-1,1),y_pred_val))
    a5 = f1_score(y_te.values.reshape(-1,1),y_pred_val)
    accuracies.append(a1)
    clfreport.append(a2)
    rascores.append(a3)
    mcc.append(a4)
    f1.append(a5)

    i = i+1
    
print("--> Execution Time: %s seconds" % (time.time() - start_time))


# In[ ]:


print("--> Mean Accuracy:",np.mean(accuracies))


# In[ ]:


print("--> Mean ROC_AUC Score:",np.mean(rascores))


# In[ ]:


print("--> Mean F1 Score:",np.mean(f1))

