import numpy as np
import pandas as pd
import math

alpha = 0.00001
thetas = [1, 2, 3, 4, 5, 6]


train_data = pd.read_csv('/home/aadi/Desktop/ML/train.csv')
test_data= pd.read_csv('/home/aadi/Desktop/ML/test.csv')
gender_data = pd.read_csv('/home/aadi/Desktop/ML/gender_submission.csv')

train_f = train_data.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'],axis=1)
test_f = test_data.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'],axis=1)
gender_f = gender_data.drop(['PassengerId'],axis=1)

train_f['Age'] = train_f['Age'].fillna(train_f['Age'].mean())
#print(train_f)

test_f['Age'] = test_f['Age'].fillna(test_f['Age'].mean())


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
    blanklist.append(1)
    blanklist.append(train_list[i][0])
    blanklist.append(train_list[i][1])
    blanklist.append(train_list[i][3])
    blanklist.append(train_list[i][4])
    blanklist.append(train_list[i][5])
    blanklist.append(train_list[i][2])
    datalist.append(blanklist)

#print(datalist)
    
    
    
datalisttest = []
for i in range(0,len(test_list)):
    blank = []
    blank.append(1) 
    blank.append(test_list[i][5]) 
    blank.append(test_list[i][0])
    blank.append(test_list[i][2])
    blank.append(test_list[i][3])
    blank.append(test_list[i][4])
    blank.append(test_list[i][1])         
    datalisttest.append(blank)
    
#print(datalisttest)    
    



for iternumber in range(0,300000):

    Dels = [0, 0, 0, 0, 0, 0]
    CostFun = 0

    for data in datalist[0:len(datalist)]:
        for j in range(0,6):
            Dels[j] += (2*(thetas[0]*data[0] + thetas[1]*data[1] + thetas[2]*data[2] +
                           thetas[3]*data[3] + thetas[4]*data[4] +thetas[5]*data[5] -data[6] )*data[j])

    for j in range(0,6):
        Dels[j] = Dels[j]/len(datalist)

    for j in range(0,6):
        thetas[j] = thetas[j] - alpha*Dels[j]

    for data in datalist[0:len(datalist)]:
        CostFun += (math.pow((thetas[0]*data[0]+ thetas[1]*data[1] + thetas[2]*data[2] +thetas[3]*data[3] + thetas[4]*data[4] + thetas[5]*data[5] - data[6]),2))

    CostFun = CostFun/len(datalist)

    print('The value of Cost Function in iteration number '+str(iternumber)+" is "+str(CostFun))


#testing

for data in datalisttest[0:len(datalisttest)]:
    PredictedAge= (thetas[0]*data[0]+ thetas[1]*data[1] + thetas[2]*data[2] +thetas[3]*data[3] +
           thetas[4]*data[4] + thetas[5]*data[5])

    #Error= PredictedAge-data[6]

    print('The value of age='+str(data)+'    is  '+ str(PredictedAge))
    #print('The value of age='+str(data)+'    is  '+ str(Error))
    


































