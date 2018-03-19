import numpy as np
import pandas as pd
import math
from scipy import stats




train_data = pd.read_csv('/home/aadi/Desktop/ML/train.csv')
test_data= pd.read_csv('/home/aadi/Desktop/ML/test.csv')
gender_data = pd.read_csv('/home/aadi/Desktop/ML/gender_submission.csv')

train_f = train_data.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'],axis=1)
test_f = test_data.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'],axis=1)
gender_f = gender_data.drop(['PassengerId'],axis=1)

train_f['Age'] = train_f['Age'].fillna(train_f['Age'].mean())
#print(train_f)

test_f['Age'] = test_f['Age'].fillna(test_f['Age'].mean())
#print(test_f)

#print(gender_f)

#print(type(train_f))
#print(type(test_f))
#print(type(gender_f)

test_f = pd.concat([test_f, gender_f], axis =1)

#print(test_f)



fortrain_ndarray = np.array(train_f)
#print(fortrain_ndarray)

fortest_ndarray = np.array(test_f)
#print(fortest_ndarray)



train_list = fortrain_ndarray.tolist()
#print(train_list)


test_list = fortest_ndarray.tolist()

datalist = []
for i in range(0,len(train_list)):
    blanklist = []
    blanklist.append(train_list[i][1])
    blanklist.append(train_list[i][2])
    blanklist.append(train_list[i][3])
    blanklist.append(train_list[i][4])
    blanklist.append(train_list[i][5])
    blanklist.append(train_list[i][0])
    datalist.append(blanklist)

#print(datalist)
    
    
    
datalisttest = []
for i in range(0,len(test_list)):
    blanl = []
    blanl.append(test_list[i][0]) 
    blanl.append(test_list[i][1])
    blanl.append(test_list[i][2])
    blanl.append(test_list[i][3])
    blanl.append(test_list[i][4])
    blanl.append(test_list[i][5])         
    datalisttest.append(blanl)
    
#print(datalisttest)    
    

#Going to calculate MVU Estimates of Natural Parameters of Gaussian Distribution of Mean Frequency for Male and Female separately

notsurvived = []
survived = []

for Data in datalist:

    if Data[5] == 0:
        notsurvived.append(Data[0])
    else:
        survived.append(Data[0])

MVUEstimateMUNotsurMeanFreq = np.mean(notsurvived)
MVUEstimateVARNotsurMeanFreq = np.var(notsurvived)

MVUEstimateMUSurMeanFreq = np.mean(survived)
MVUEstimateVARSurMeanFreq = np.var(survived)



print(notsurvived)

TestingData = []


#Going to perform testing of Bayes Classifier
CorrectCount = 0

for Data in datalisttest:

    PosteriorNotSurvived = stats.norm.pdf(Data[0],MVUEstimateMUNotsurMeanFreq,math.sqrt(MVUEstimateMUNotsurMeanFreq))
    PosteriorSurvived = stats.norm.pdf(Data[0],MVUEstimateVARNotsurMeanFreq,math.sqrt(MVUEstimateVARNotsurMeanFreq))

    PriorNotSurvived = (PosteriorNotSurvived)/(PosteriorNotSurvived + PosteriorSurvived)
    PriorSurvived = (PosteriorSurvived)/(PosteriorNotSurvived + PosteriorSurvived)

    if PriorNotSurvived > PriorSurvived and Data[5] == 0:
        CorrectCount += 1

    elif PriorNotSurvived < PriorSurvived and Data[5] == 1:
        CorrectCount += 1



print("The accuracy of our implemented Notsurvived/survived  Bayes Classifier for single feature is " + str(float(CorrectCount/(len(datalisttest)))))












