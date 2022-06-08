#### performed by:
#### Name: Ashish Sharma
#### Reference Number: 201604257 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from warnings import filterwarnings
filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report,f1_score,recall_score,precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

"""

        Need to Import all the above mentioned libraries to perform the program in the best possible way :)

"""

data=pd.read_csv('C:\library\diabetes.csv')
data.head()
#print(data.head())
#print(data.shape)
#print(data.info())
#print(data.describe())

############################################################################################################
##################################### Decision Tree ########################################################
############################################################################################################




pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)


data=pd.read_csv('C:\library\diabetes.csv')

data.head()
#print(data.head())

data.describe()
#print(data.describe())

X= data.drop("Outcome",axis=1)
y=data["Outcome"]

X_train = X.iloc[:600]
X_test = X.iloc[600:]
y_train = y[:600]
y_test = y[600:]

print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)
print("y_train Shape: ",y_train.shape)
print("y_test Shape: ",y_test.shape)

Decision_Tree=DecisionTreeClassifier().fit(X_train,y_train)

#print(Decision_Tree)

Decision_Tree.get_params()
#print(Decision_Tree.get_params())

Decision_Tree.get_n_leaves()
#print(Decision_Tree.get_n_leaves())

Decision_Tree.get_depth()
#print(Decision_Tree.get_depth())


plt.figure(figsize=(15,10))
plot_tree(Decision_Tree,proportion=True)
plt.show()


###############
##Predicition##
###############

y_predict=Decision_Tree.predict(X_test)
ConfusionMatrix=confusion_matrix(y_test,y_predict)

#print(ConfusionMatrix)

print("Our Accuracy is: ",(ConfusionMatrix[0][0]+ConfusionMatrix[1][1])/(ConfusionMatrix[0][0]+ConfusionMatrix[0][1]+ConfusionMatrix[1][0]+ConfusionMatrix[1][1]))

accuracy_score(y_test,y_predict)
recall_score(y_test,y_predict)
precision_score(y_test,y_predict)
f1_score(y_test,y_predict)
print(classification_report(y_test,y_predict))


###################
## Model Tuning##
###################

Accuracies=cross_val_score(estimator=Decision_Tree,X=X_train,y=y_train,cv=10)
print("Average Accuracies: {:.2f}%".format(Accuracies.mean()*100))
print("Standard Deviation of Accuracies: {:.2f}%".format(Accuracies.std()*100))

Decision_Tree.predict(X_test)[:10]

Decision_Tree_Params={'criterion':["gini","entropy"],'max_depth':[1,2,3,4,5,6,7,8,9,10],'min_samples_split':list(range(1,10))}

Decision_Tree_Classifier=DecisionTreeClassifier()
Decision_Tree_CV=GridSearchCV(Decision_Tree_Classifier,Decision_Tree_Params,cv=9,n_jobs=-1,verbose=2)

Start_time=time.time()

Decision_Tree_CV.fit(X_train,y_train)

Elapsed_Time=time.time() - Start_time

print(f"Elapsed time for the Decision Tree Classification Cross Validation is: "f"{Elapsed_Time:.3f}seconds")

Decision_Tree_CV.best_score_
#print(Decision_Tree_CV.best_score_)

Decision_Tree_CV.best_params_
'''
The command below will help in understanding the best parameters
'''
#print(Decision_Tree_CV.best_params_)

Decision_Tree_Tuned=DecisionTreeClassifier(criterion="entropy",max_depth=2,min_samples_split=2).fit(X_train,y_train)

Decision_Tree_Tuned

y_predict=Decision_Tree_Tuned.predict(X_test)

ConfusionMatrix=confusion_matrix(y_test,y_predict)

#print(ConfusionMatrix)

print("Our Accuracies is: ",(ConfusionMatrix[0][0]+ConfusionMatrix[1][1])/(ConfusionMatrix[0][0]+ConfusionMatrix[1][0]+ConfusionMatrix[0][1]+ConfusionMatrix[1][1]))

accuracy_score(y_test,y_predict)

precision_score(y_test,y_predict)

f1_score(y_test,y_predict)

print(classification_report(y_test,y_predict))



###############################################################################################################
##################################### Random_Forest ###########################################################
###############################################################################################################



pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)


data=pd.read_csv('C:\library\diabetes.csv')
data.head()

X=data.drop("Outcome",axis=1)
y=data["Outcome"]

X_train= X.iloc[:600]
X_test=X.iloc[600:]
y_train=y[:600]
y_test=y[600:]

print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)
print("y_train Shape: ",y_train.shape)
print("y_test Shape: ",y_test.shape)

Random_Forest= RandomForestClassifier().fit(X_train,y_train)

Random_Forest.get_params()

feature_importances=pd.DataFrame({"Features":X_train.columns,"Feature Importances":Random_Forest.feature_importances_}).sort_values(by="Feature Importances")

feature_importances.head()

plt.figure(figsize=(15,7),dpi=200)
sns.barplot(data=feature_importances,x="Features",y="Feature Importances")
plt.title("Feature Importances")
plt.xticks(rotation=90)
plt.show()

feature_importances["Feature Importances"].cumsum()

Random_Forest

y_pred= Random_Forest.predict(X_test)
ConfuseMatrix=confusion_matrix(y_test,y_pred)

print("Our Accuracy is: ",(ConfuseMatrix[0][0]+ConfuseMatrix[1][1])/(ConfuseMatrix[0][0]+ConfuseMatrix[1][1]+ConfuseMatrix[0][1]+ConfuseMatrix[1][0]))

accuracy_score(y_test,y_pred)

recall_score(y_test,y_pred)

precision_score(y_test,y_pred)

f1_score(y_test,y_pred)

print(classification_report(y_test,y_pred))



###################
## Model Tuning##
###################


Accuracies= cross_val_score(estimator=Random_Forest,X=X_train,y=y_train,cv=10)

print("Average Accuracy: {:.2f} %".format(Accuracies.mean()*100))
print("Standard Deviation of Accuracies: {:.2f} %".format(Accuracies.std()*100))

Random_Forest.predict(X_test)[:10]

Random_Forests_params = {'max_depth': list(range(1,10)),'max_features':[2,5,7,8],'n_estimators':[300,500,1000,1700,2000],'criterion':["gini","entropy"]}

Random_Forest_classifier=RandomForestClassifier()
Random_Forest_cv= GridSearchCV(Random_Forest_classifier,Random_Forests_params,cv=9,n_jobs=-1,verbose=2)

start_time=time.time()

Random_Forest_cv.fit(X_train,y_train)

Elapsed_time=time.time() - start_time

print(f"Elapsed time for Random Forests Classifier cross validation: "f"{Elapsed_time:.3f} seconds")

Random_Forest_cv.best_score_
#print(Random_Forest_cv.best_score_)
Random_Forest_cv.best_params_
'''
The command below will help in understanding the best parameters
'''

#print(Random_Forest_cv.best_params_)

Random_Forest_tuned= RandomForestClassifier(criterion="gini",max_depth=5,max_features=2,n_estimators=500).fit(X_train,y_train)

y_pred=Random_Forest_tuned.predict(X_test)

ConfuseMatrix=confusion_matrix(y_test,y_pred)


print("Our Accuracy is: ",(ConfuseMatrix[0][0]+ConfuseMatrix[1][1])/(ConfuseMatrix[0][0]+ConfuseMatrix[1][1]+ConfuseMatrix[0][1]+ConfuseMatrix[1][0]))

accuracy_score(y_test,y_pred)

recall_score(y_test,y_pred)

precision_score(y_test,y_pred)

f1_score(y_test,y_pred)

print(classification_report(y_test,y_pred))

Importances=pd.DataFrame({"Importance":Random_Forest_tuned.feature_importances_*100},index=X_train.columns)

Importances.head()

Importances.sort_values(by="Importance",axis=0,ascending=True).plot(kind="barh",color="b")
plt.xlabel("Feature Importances")
plt.ylabel("Features")
plt.title("Feature Importances")
plt.show()

##################################################################################################
###################################### KNN Model #################################################
##################################################################################################

sns.pairplot(data,hue='Outcome')
#plt.show()

cor=data.corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.CMRmap_r)
#plt.show()
sns.set_theme(style='whitegrid')
sns.boxplot(x="Age",data=data,palette="Set3")
plt.title("Age Distribution")
#plt.show()

figre=plt.figure(figsize=(15,20))
ax=figre.gca()
data.hist(ax=ax) 
#plt.show()

data.Outcome.value_counts().plot(kind='bar')
plt.xlabel("Diabetes or Not")
plt.ylabel("Count")
plt.title("Output")
#plt.show()

X=data.drop('Outcome',axis=1)
X.head()
#print(X.head())

y=data['Outcome']
y.head()
#print(y.head())

X_train,X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)

#print(X_train.shape)

sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)

#print(X_train)
#print(X_test)

Knn=KNeighborsClassifier(n_neighbors=5,metric='euclidean',p=2)
Knn.fit(X_train,y_train)
#print(Knn.fit(X_train,y_train))

y_pred=Knn.predict(X_test)
y_pred
#print(y_pred)


Knn.score(X_test,y_test)
#print(Knn.score(X_test,y_test))

metrics.accuracy_score(y_test,y_pred)
#print(metrics.accuracy_score(y_test,y_pred))

FinalMatrix=confusion_matrix(y_test,y_pred)
FinalMatrix
#print(FinalMatrix)

Target_Names=['Diabetes','Normal']
#print(classification_report(y_test,y_pred,target_names=Target_Names))

#For selecting K value

Error_Rate=[]

for i in range(1,50):
    Knn=KNeighborsClassifier(n_neighbors=i)
    Knn.fit(X_train,y_train)
    pred_i=Knn.predict(X_test)
    Error_Rate.append(np.mean(pred_i !=y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,50),Error_Rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate V/S K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()


#From graph we can see that optimize k value is 16,17,18
# Now we will train our KNN classifier with this k values

Knn=KNeighborsClassifier(n_neighbors=18,metric='euclidean',p=2)
Knn.fit(X_train,y_train)
#print(Knn.fit(X_train,y_train))

y_pred=Knn.predict(X_test)
#print(y_pred)

Knn.score(X_test,y_test)

FinalMatrix=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(FinalMatrix,annot=True)
#plt.show()

Target_Names=['Diabetes','Normal']
print(classification_report(y_test,y_pred,target_names=Target_Names))





#####################################################################################################
########################### Choosing The Best Model #################################################
#####################################################################################################

model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = KNeighborsClassifier()

"""
    The models are divided and renamed as Model1,2 and 3 which will further help in visual representation of data
    and providing the final Confusion Matrix
"""

model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
model3.fit(X_train,y_train)


predict_prob1_ts = model1.predict_proba(X_test)
predict_prob2_ts = model2.predict_proba(X_test)
predict_prob3_ts = model3.predict_proba(X_test)

predict_prob1_tr = model1.predict_proba(X_train)
predict_prob2_tr = model2.predict_proba(X_train)
predict_prob3_tr = model3.predict_proba(X_train)


fpr1_ts, tpr1_ts, thresh1_ts = roc_curve(y_test, predict_prob1_ts[:,1], pos_label=1)
fpr2_ts, tpr2_ts, thresh2_ts = roc_curve(y_test, predict_prob2_ts[:,1], pos_label=1)
fpr3_ts, tpr3_ts, thresh3_ts = roc_curve(y_test, predict_prob3_ts[:,1], pos_label=1)

fpr1_tr, tpr1_tr, thresh1_tr = roc_curve(y_train, predict_prob1_tr[:,1], pos_label=1)
fpr2_tr, tpr2_tr, thresh2_tr = roc_curve(y_train, predict_prob2_tr[:,1], pos_label=1)
fpr3_tr, tpr3_tr, thresh3_tr = roc_curve(y_train, predict_prob3_tr[:,1], pos_label=1)


random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)



auc_score1 = roc_auc_score(y_test, predict_prob1_ts[:,1])
auc_score2 = roc_auc_score(y_test, predict_prob2_ts[:,1])
auc_score3 = roc_auc_score(y_test, predict_prob3_ts[:,1])


print(f"{auc_score1} , {auc_score2}, {auc_score3}")

plt.figure(figsize=(16, 10))
plt.plot(fpr1_ts, tpr1_ts, linestyle='--',color='r', label='KNN Model Test')
plt.plot(fpr2_ts, tpr2_ts, linestyle=':',color='lime', label='RandomForest Test')
plt.plot(fpr3_ts, tpr3_ts, linestyle='-.',color='b', label='DecisionTree Test')

plt.plot(fpr1_tr, tpr1_tr, linestyle='--',color='pink', label='KNN Model Train')
plt.plot(fpr2_tr, tpr2_tr, linestyle=':',color='orange', label='RandomForest Train')
plt.plot(fpr3_tr, tpr3_tr, linestyle='-.',color='dodgerblue', label='DecisionTree Train')

plt.plot(p_fpr, p_tpr, linestyle='-', color='y')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()


final_model = KNeighborsClassifier()
final_model.fit(X_train, y_train)
finalpred = final_model.predict(X_test)
print(f"Classification report\n {classification_report(finalpred, y_test)}")
print(f"Score = {final_model.score(X_test,y_test)}")