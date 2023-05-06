from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
 

df = pd.read_csv('train.csv')
df.drop_duplicates(inplace=True)
# print(df.describe()) 

# print(df.isna().sum().sort_values(ascending=False))
# Cabin        687
# Age          177

FeaturesToConvert = ['Sex', 'Embarked']
le = LabelEncoder()
for feature in FeaturesToConvert:
    df[feature] = le.fit_transform(df[feature])
    
df.drop(['Name'], axis=1, inplace=True)
df.drop(['Cabin'], axis=1, inplace=True)
df.dropna(subset=['Embarked'], inplace=True)

X_num = df[['Age', 'SibSp', 'Parch', 'Fare']].values
X_cat = df[['Pclass', 'Sex', 'Embarked']].values
y = df['Survived'].values

# steps = [('imputation', SimpleImputer(strategy='mean')), 
#          ('scaler', StandardScaler())
#          ('logistic_regression', LogisticRegression())]

X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y, test_size=0.1, random_state=42)
X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y, test_size=0.1, random_state=42)

imp_cat = SimpleImputer(strategy='most_frequent')
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)

imp_num = SimpleImputer(strategy='median')
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)

scaler = MinMaxScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

X_train = np.append(X_train_cat, X_train_num, axis=1)
X_test = np.append(X_test_cat, X_test_num, axis=1)

# models = {"Logistic Regression": LogisticRegression(), 
#           "KNN": KNeighborsClassifier(), 
#           "Decision Tree": DecisionTreeClassifier(), 
#           "Random Forest": RandomForestClassifier()}
# results = []
# for model in models.values():
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
#     results.append(cv_results)
    
# plt.boxplot(results, labels=models.keys())
# plt.title('Algorithm Comparison')
# plt.show()

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     print(name, model.score(X_test, y_test))

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# parameters = {
#     'penalty' : ['l2'], 
#     'C'       : np.logspace(-3,3,7),
#     'solver'  : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],}
# log_reg_CV = GridSearchCV(LogisticRegression(), param_grid=parameters, scoring='accuracy', cv=kf)
# log_reg_CV.fit(X_train, y_train) 
# print(log_reg_CV.best_params_, log_reg_CV.best_score_)

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# parameters = {
#     'criterion' : ['gini', 'entropy', 'log_loss'],
#     'bootstrap': [True, False],
#     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#     'min_samples_leaf': [1, 2, 4],
#     'min_samples_split': [2, 5, 10]
# }
# log_reg_CV = GridSearchCV(RandomForestClassifier(), param_grid=parameters, scoring='accuracy', cv=kf)

model = RandomForestClassifier(bootstrap=True, criterion='entropy', max_depth=20, min_samples_leaf=4, min_samples_split=2)
model.fit(X_train, y_train) 
# {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 2} 0.8347096774193549
# print(log_reg_CV.best_params_, log_reg_CV.best_score_)
print(model.score(X_test, y_test))


##############################################
df = pd.read_csv('train.csv')

FeaturesToConvert = ['Sex', 'Embarked']
le = LabelEncoder()
for feature in FeaturesToConvert:
    df[feature] = le.fit_transform(df[feature])
    
df.drop(['Name'], axis=1, inplace=True)
df.drop(['Cabin'], axis=1, inplace=True)
df.dropna(subset=['Embarked'], inplace=True)

X_num = df[['Age', 'SibSp', 'Parch', 'Fare']].values
X_cat = df[['Pclass', 'Sex', 'Embarked']].values
y = df['Survived'].values

X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y, test_size=1, random_state=42)
X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y, test_size=1, random_state=42)

imp_cat = SimpleImputer(strategy='most_frequent')
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)

imp_num = SimpleImputer(strategy='median')
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)

scaler = MinMaxScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

X_train = np.append(X_train_cat, X_train_num, axis=1)
X_test = np.append(X_test_cat, X_test_num, axis=1)

model.score(X_test, y_test)