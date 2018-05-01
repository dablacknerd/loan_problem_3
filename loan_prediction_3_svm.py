import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from categorical_encoder import CategoricalEncoder,DataFrameSelector
from sklearn.pipeline import FeatureUnion

filr_train = os.path.join(os.getcwd(),'data','loan_problem_3.csv')
df = pd.read_csv(filr_train)
df1 = df.dropna()

filr_test = os.path.join(os.getcwd(),'data','test.csv')
df_test = pd.read_csv(filr_test)

cat_features = ['Gender','Married','Education','Self_Employed','Property_Area','Dependents']
num_features = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_features)),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_features)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

X_train = full_pipeline.fit_transform(df1)
x_test = full_pipeline.fit_transform(df_test)
y,y_cats = df1['Loan_Status'].factorize()

svm_model_rbf = svm.SVC(kernel='rbf')
svm_model_rbf.fit(X_train,y)
predictions = svm_model_rbf.predict(x_test)

solution_filr = os.path.join(os.getcwd(),"data","solution_submission.csv")
loan_ids = df_test['Loan_ID'].values

results =[]

for i in range(0,len(predictions)):
    if predictions[i] == 1:
        results.append([loan_ids[i],'Y'])
    else:
        results.append([loan_ids[i],'N'])

final_results = pd.DataFrame(results,columns=['Loan_ID','Loan_Status'])
final_results.to_csv(solution_filr,index=False)
