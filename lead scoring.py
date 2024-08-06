#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


new_directory ='/Users/pankajyadav/Downloads/Lead Scoring Assignment'
os.chdir(new_directory)


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
import time
import warnings
warnings.filterwarnings('ignore')


# In[4]:


df = pd.read_csv('Leads.csv')


# In[5]:


df.info()


# In[6]:


df.shape


# ### Dealing with missing values

# In[7]:


df.isna().sum()


# In[8]:


df.isna().sum()/df.shape[0]*100


# In[9]:


df = df.drop(columns = ['Prospect ID','Lead Number'])


# In[10]:


categorical = ['Lead Origin','Lead Source','Do Not Email','Do Not Call','Last Activity','Country','Specialization', 'How did you hear about X Education',
            'What is your current occupation','What matters most to you in choosing a course','Search',
            'Magazine','Newspaper Article','Digital Advertisement', 'Through Recommendations','Receive More Updates About Our Courses',
            'Update me on Supply Chain Content','Lead Profile','City','Asymmetrique Activity Index','Newspaper','X Education Forums',
             'I agree to pay the amount through cheque','Tags','Lead Quality','Asymmetrique Profile Index',
            'A free copy of Mastering The Interview','Last Notable Activity','Get updates on DM Content']
numerical=['TotalVisits','Total Time Spent on Website','Page Views Per Visit','Asymmetrique Activity Score','Asymmetrique Profile Score']


# In[11]:


for cat in categorical:
    df[cat]= df[cat].fillna(df[cat].mode()[0])
for num in numerical :
    df[num]= df[num].fillna(df[num].median())


# In[12]:


df.isna().sum()


# In[13]:


df[categorical].nunique().sort_values()


# In[14]:


#Identifying redundant features, which have only one level
x = df[categorical].nunique()
cols_to_drop = list(x[x < 2].index)
cols_to_drop


# In[15]:


df = df.drop(columns = ['Magazine','Receive More Updates About Our Courses','Update me on Supply Chain Content', 'Get updates on DM Content','I agree to pay the amount through cheque'])


# In[16]:


categorical = ['Lead Origin','Lead Source','Do Not Email','Do Not Call','Last Activity','Country','Specialization', 'How did you hear about X Education',
            'What is your current occupation','What matters most to you in choosing a course','Search',
            'Newspaper Article','Digital Advertisement', 'Through Recommendations',
            'Lead Profile','City','Asymmetrique Activity Index','X Education Forums','Newspaper',
             'Tags','Lead Quality','Asymmetrique Profile Index',
            'A free copy of Mastering The Interview','Last Notable Activity']

#binary catagorical variables
x = df[categorical].nunique()
binary_cat_cols = list(x[x == 2].index)
binary_cat_cols


# In[17]:


for k in binary_cat_cols:    
    print('{}\n'.format(df[k].value_counts()))


# In[18]:


df = df.drop(columns = ['Do Not Call','Newspaper Article','Digital Advertisement','X Education Forums','Newspaper'] )


# In[19]:


df['Country'].value_counts(dropna = False, normalize = True).mul(100).round(2)


# ##### Almost all the values in this column are India so this is not usefull 

# In[20]:


df['Lead Source'].value_counts(dropna = False, normalize = True).mul(100).round(2)


# In[21]:


#replacing google with Google
df['Lead Source'] = df['Lead Source'].replace('google','Google')

#combining less frequent levels into one, 'Others'
x = df['Lead Source'].value_counts(normalize = True).mul(100)
df['Lead Source'] = df['Lead Source'].replace(list(x[x < 1].index), 'Others')
df['Lead Source'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [10,5])
plt.xticks(rotation = 45)


# In[22]:


df['Last Activity'].value_counts(dropna = False, normalize = True).mul(100).round(2)


# In[23]:


#combining less frequent levels into one, 'Others'
x = df['Last Activity'].value_counts(normalize = True).mul(100)
df['Last Activity'] = df['Last Activity'].replace(list(x[x < 2].index), 'Others')
df['Last Activity'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [10,5])
plt.xticks(rotation = 45)


# In[24]:


df['Specialization'].value_counts(dropna = False, normalize = True).mul(100).round(2)


# In[25]:


#combining less frequent levels into one, 'Others'
x = df['Specialization'].value_counts(normalize = True).mul(100)
df['Specialization'] = df['Specialization'].replace(list(x[x < 2].index), 'Others')
df['Specialization'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [10,5])
plt.xticks(rotation = 90)


# In[26]:


df['Tags'].value_counts(dropna = False, normalize = True).mul(100).round(2)


# In[27]:


#combining less frequent levels into one, 'Others'
x = df['Tags'].value_counts(normalize = True).mul(100)
df['Tags'] = df['Tags'].replace(list(x[x < 2].index), 'Others')
df['Tags'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [10,5])
plt.xticks(rotation = 90)


# In[28]:


df['Last Notable Activity'].value_counts(dropna = False, normalize = True).mul(100).round(2)


# In[29]:


#combining less frequent levels into one, 'Others'
x = df['Last Notable Activity'].value_counts(normalize = True).mul(100)
df['Last Notable Activity'] = df['Last Notable Activity'].replace(list(x[x < 2].index), 'Others')
df['Last Notable Activity'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [10,5])
plt.xticks(rotation = 90)


# In[30]:


label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])


# In[31]:


categorical = ['Lead Origin','Lead Source','Do Not Email','Last Activity','Specialization', 'How did you hear about X Education',
            'What is your current occupation','What matters most to you in choosing a course','Search',
             'Through Recommendations',
            'Lead Profile','City','Asymmetrique Activity Index',
             'Tags','Lead Quality','Asymmetrique Profile Index',
            'A free copy of Mastering The Interview','Last Notable Activity']
for column in categorical:
    unique_values = df[column].nunique()
    print(f"Unique values in column '{column}': {unique_values}")


# In[32]:


df=pd.get_dummies(data=df, drop_first=True)


# In[33]:


# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Fit and transform the data
encoded_data = encoder.fit_transform(df[categorical])

# Convert the encoded data back to DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical))

# Merge encoded data with original DataFrame
df1 = pd.concat([df, encoded_df], axis=1)

# Drop original categorical columns
df1.drop(categorical, axis=1, inplace=True)


# In[34]:


#df1.head(10)


# In[35]:


df1.columns


# In[36]:


df1.head()


# ## Split the dataset into two datasets with Converted=1 (leads which are coverted) and Converted=0 (all other cases) 

# ## We will use these two datasets for a small number of comparisons

# In[37]:


converted0 = df1[df1["Converted"]==0]
converted1 = df1[df1["Converted"]==1]


# In[38]:


total = len(df1["Converted"])
explode = [0, 0.05]

def my_fmt(x):
    return '{:.2f}%\n({:.0f})'.format(x, total*x/100) #to print both the percentage and value together

plt.figure(figsize = [6, 6])
plt.title("Imbalance between converted0 and converted1")
df["Converted"].value_counts().plot.pie(autopct = my_fmt, colors = ["teal", "gold"], explode = explode)

plt.show()


# #### From above Graph it is clear that only 38.54% of the leads are converted whereas 61.46% are not converted

# In[39]:


x = df1.drop("Converted", axis = 1)
y = df1["Converted"]


# In[40]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)


# ## Logistic Regression

# In[41]:


start_time = time.time()
lr1 = LogisticRegression()
lr1.fit(x_train, y_train)
end_time = time.time()
y_pred = lr1.predict(x_train)
accuracy = accuracy_score(y_train, y_pred)
y_pred1 = lr1.predict(x_test)
accuracy = accuracy_score(y_test, y_pred1)
print(f'{lr1.__class__.__name__} Test : {accuracy:.3f}')
runtime = end_time - start_time
print(f"Model training time: {runtime} seconds" )


# #### Grid Search

# In[42]:


# Range of max_depth values to explore
param_grid = {'C': 10**np.linspace(-3,3,20),
             'penalty' : ['l1','l2']}
lr2 = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
lr_gridsearch = GridSearchCV(lr2, param_grid, cv=7, scoring='accuracy', refit=True)
lr_gridsearch = lr_gridsearch.fit(x, y)
lr2.fit(x_train,y_train)
best_model = lr_gridsearch.best_estimator_
print(classification_report(y_test,best_model.predict(x_test)))
print(lr_gridsearch.best_params_)
# Calculate the test accuracy of the best model
test_accuracy = accuracy_score(y_test, best_model.predict(x_test))
print("Test Accuracy:", test_accuracy)


# #### Cross Validation

# In[43]:


# Perform 9-fold cross-validation
lr3 = cross_val_score(LogisticRegression(), x, y, cv=9)
# Print the cross-validation scores
print("Cross-validation scores:", lr3)
# Calculate and print the mean and standard deviation of the cross-validation scores
print("Mean accuracy:", lr3.mean())
print("Standard deviation of accuracy:", lr3.std())


# ## Decision Tree Classifier

# In[44]:


start_time = time.time()
dt1 = DecisionTreeClassifier()
dt1.fit(x_train, y_train)
end_time = time.time()
y_pred = dt1.predict(x_train)
accuracy = accuracy_score(y_train, y_pred)
y_pred1 = dt1.predict(x_test)
accuracy = accuracy_score(y_test, y_pred1)
print(f'{dt1.__class__.__name__} Test : {accuracy:.3f}')
runtime = end_time - start_time
print(f"Model training time: {runtime} seconds" )


# #### Hyperparameter Tuning

# In[45]:


# Range of max_depth values to explore
max_depth_range = range(1, 21)
# Lists to store accuracies
train_accuracies = []
test_accuracies = []
# Looping through the range of max_depth values
for depth in max_depth_range:
    # Create and fit the model
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(x_train, y_train)
    # Predict on training and testing sets
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    # Calculate and append accuracies
    train_accuracies.append(accuracy_score(y_train, y_train_pred))
    test_accuracies.append(accuracy_score(y_test, y_test_pred))
# Plotting
plt.figure(figsize=(7, 4))
plt.plot(max_depth_range, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(max_depth_range, test_accuracies, label='Test Accuracy', marker='x')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy vs Max Depth')
plt.legend()
plt.grid(True)
plt.show()


# In[46]:


# Define the parameter grid for the grid search
param_grid = {
    'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9]}

# Perform the grid search
dt2 = GridSearchCV(DecisionTreeClassifier(),param_grid, cv=5, scoring='accuracy')
start_time = time.time()
dt2.fit(x_train, y_train)
end_time = time.time()
# Print the best parameters and the corresponding score

print(dt2.best_params_)
runtime = end_time - start_time
print(f"Model training time: {runtime} seconds" )
best_model = dt2.best_estimator_
print(classification_report(y_test, best_model.predict(x_test)))
# Calculate the training accuracy of the best model
train_accuracy = accuracy_score(y_train, best_model.predict(x_train))
print("Training Accuracy:", train_accuracy)

# Calculate the test accuracy of the best model
test_accuracy = accuracy_score(y_test, best_model.predict(x_test))
print("Test Accuracy:", test_accuracy)


# #### Cross Validation for Decision Tree

# In[47]:


# Perform 9-fold cross-validation
scores = cross_val_score(DecisionTreeClassifier(), x, y, cv=9)

# Print the cross-validation scores
print("Cross-validation scores:", scores)

# Calculate and print the mean and standard deviation of the cross-validation scores
print("Mean accuracy:", scores.mean())
print("Standard deviation of accuracy:", scores.std())


# ## Random Forest Classifier

# In[48]:


start_time = time.time()
rf1 = RandomForestClassifier()
rf1.fit(x_train, y_train)
end_time = time.time()
y_pred = rf1.predict(x_train)
accuracy = accuracy_score(y_train, y_pred)
y_pred1 = rf1.predict(x_test)
accuracy = accuracy_score(y_test, y_pred1)
print(f'{rf1.__class__.__name__} Test : {accuracy:.3f}')
runtime = end_time - start_time
print(f"Model training time: {runtime} seconds" )


# #### Hyperparameter Tuning

# In[49]:


# Range of max_depth values to explore
max_depth_range = range(1, 21)
# Lists to store accuracies
train_accuracies = []
test_accuracies = []
# Looping through the range of max_depth values
for depth in max_depth_range:
    # Create and fit the model
    model = RandomForestClassifier(max_depth=depth)
    model.fit(x_train, y_train)
    # Predict on training and testing sets
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    # Calculate and append accuracies
    train_accuracies.append(accuracy_score(y_train, y_train_pred))
    test_accuracies.append(accuracy_score(y_test, y_test_pred))
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(max_depth_range, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(max_depth_range, test_accuracies, label='Test Accuracy', marker='x')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy vs Max Depth')
plt.legend()
plt.grid(True)
plt.show()


# In[50]:


# Define the parameter grid for the grid search
param_grid = {
    'max_depth': [8,9,10,11,12],
    'max_features': ['auto','sqrt','log2'],
    'min_samples_leaf': [3, 4, 5, 6],
    'n_estimators': [50, 100, 200, 300, 400, 500]
}

# Perform the grid search
grid_search = GridSearchCV(RandomForestClassifier(),param_grid, cv=7, scoring='accuracy')
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
best_model = grid_search.best_estimator_
print(classification_report(y_test, best_model.predict(x_test)))

# Calculate the test accuracy of the best model
test_accuracy = accuracy_score(y_test, best_model.predict(x_test))
print("Test Accuracy:", test_accuracy)


# In[51]:


rf2 = RandomForestClassifier(max_depth= 12, max_features= 'sqrt', min_samples_leaf= 3, n_estimators= 500)
start_time = time.time()
rf2.fit(x_train,y_train)
end_time = time.time()
pred2 = rf2.predict(x_test)
print('Test Accuracy:',accuracy_score(y_test,pred2))

runtime = end_time - start_time
print(f"Model training time: {runtime} seconds" )


# #### Cross Validation

# In[52]:


# Perform 9-fold cross-validation
scores = cross_val_score(RandomForestClassifier(), x, y, cv=9)

# Print the cross-validation scores
print("Cross-validation scores:", scores)

# Calculate and print the mean and standard deviation of the cross-validation scores
print("Mean accuracy:", scores.mean())
print("Standard deviation of accuracy:", scores.std())


# In[53]:


y_pred_proba_lr = lr2.predict_proba(x_test)[:,1]
y_pred_proba_dt = dt2.predict_proba(x_test)[:,1]
y_pred_proba_rf = rf2.predict_proba(x_test)[:,1]

fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred_proba_lr)
fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_proba_dt)
fpr3, tpr3, thresholds3 = roc_curve(y_test, y_pred_proba_rf)

auc1 = roc_auc_score(y_test, y_pred_proba_lr)
auc2 = roc_auc_score(y_test, y_pred_proba_dt)
auc3 = roc_auc_score(y_test, y_pred_proba_rf)

plt.figure(figsize=(5, 5))
plt.plot(fpr1, tpr1, color='blue', lw=2, label='ROC curve Linear_reg (AUC1 = %0.2f)' % auc1)
plt.plot(fpr2, tpr2, color='red', lw=2, label='ROC curve Decision_tr (AUC2 = %0.2f)' % auc2)
plt.plot(fpr3, tpr3, color='green', lw=2, label='ROC curve Random_fr (AUC3 = %0.2f)' % auc3)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




