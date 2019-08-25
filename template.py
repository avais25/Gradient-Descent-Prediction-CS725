#!/bin/python3

import numpy as np
import pickle
import pandas as pd
import datetime as dt




"""
NOTE: All functions mentioned below MUST be implemented
      All functions must be reproducible, i.e., repeated function calls with the
      same parameters must result in the same output. Look into numpy RandomState
      to achieve this.
"""

def get_feature_matrix(file_path):
    """
    file path: path to  the file assumed to be in the same format as
               either train.csv or test_features.csv in the Kaggle competition


    Return: A 2-D numpy array of size n x m where n is the number of examples in
            the file and m your feature vector size

    NOTE: Preserve the order of examples in the file
    """








    #importing csv file as dataFrame
    df = pd.read_csv(file_path, usecols=range(2,26))
    parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')#2016-03-28 13:50:00
    dfDate=pd.read_csv(file_path,parse_dates=['date'],date_parser=parser,usecols=['date'])
    hour=dfDate['date'].apply(lambda x:x.strftime('%H'))
    hour=pd.to_numeric(hour, errors='coerce')
    #hour.columns=['hour','hour']
    #hour.rename(index=str ,columns={'date': 'hour'}, inplace=True)

    month=dfDate['date'].apply(lambda x:x.strftime('%m'))
    month=pd.to_numeric(month, errors='coerce')
    #print(month)
    #month.rename(index=str , columns={'date': 'month'}, inplace=True)
    #month.columns=['month','month']
    #print(month)
    


    #df.join(hour)
   
    df=pd.concat([hour,df],axis=1)
    #df=pd.concat([month,df],axis=1)
    #test=df.groupby('date', as_index=False)['Output'].mean()
    #df=df.rename(index=str ,columns={'date': 'hour'})
    #df=pd.concat([month,df],axis=1)
    #df.rename(index=str ,columns={'date': 'month'}, inplace=True)
    #print(df)
    #converting data frame to matrix
    
    inputMatrix=df.values

    '''for i in range(inputMatrix.shape[0]):
      if inputMatrix[i,0] in (21,22,23,0,1,2,3,4,5,6):
        inputMatrix[i,0]=600
      if inputMatrix[i,0] in (8,9,10):
        inputMatrix[i,0]=1200
      if inputMatrix[i,0] in (11,16,17,18,19,20):
        inputMatrix[i,0]=1500
      if inputMatrix[i,0] in (12,13,14,15):
        inputMatrix[i,0]=1100'''

    for i in range(inputMatrix.shape[0]):
        if inputMatrix[i,0] in (23,0,1,2,3,4,5):
          inputMatrix[i,0]=50
        if inputMatrix[i,0]==6:
          inputMatrix[i,0]=57
        if inputMatrix[i,0] in (21,15):
          inputMatrix[i,0]=100
        if inputMatrix[i,0] in (12,13,10,16):
          inputMatrix[i,0]=123
        if inputMatrix[i,0]==6:
          inputMatrix[i,0]=57
        if inputMatrix[i,0]==7:
          inputMatrix[i,0]=78
        if inputMatrix[i,0]==8:
          inputMatrix[i,0]=104
        if inputMatrix[i,0]==9:
          inputMatrix[i,0]=112
        if inputMatrix[i,0]==11:
          inputMatrix[i,0]=140
        if inputMatrix[i,0]==14:
          inputMatrix[i,0]=107
        if inputMatrix[i,0]==17:
          inputMatrix[i,0]=161
        if inputMatrix[i,0]==18:
          inputMatrix[i,0]=190
        if inputMatrix[i,0]==19:
          inputMatrix[i,0]=143
        if inputMatrix[i,0]==20:
          inputMatrix[i,0]=129

        inputMatrix[i,0]=inputMatrix[i,0]*25



    '''for i in range(inputMatrix.shape[0]):
      if inputMatrix[i,0]==1:
        inputMatrix[i,0]=980
      if inputMatrix[i,0]==2:
        inputMatrix[i,0]=1010
      if inputMatrix[i,0]==3:
        inputMatrix[i,0]=970
      if inputMatrix[i,0]==3:
        inputMatrix[i,0]=990
      if inputMatrix[i,0]==5:
        inputMatrix[i,0]=920'''

    #print(dfDate)
    #returning the 2d matrix
    return inputMatrix



    #train = np.append(train, col, axis=1)
    #parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')#2016-03-28 13:50:00
    #dfDate=pd.read_csv(file_path,parse_dates=['date'],date_parser=parser,usecols=['date'])
    #df = df.assign(bias=1)
    #converting data frame to matrix
    #print(dataFrame)
    #inputMatrix=df.values
    #print(dfDate)
    #returning the 2d matrix
    #return train
    

def get_output(file_path):
    """
    file_path: path to a file in the same format as in the Kaggle competition

    Return: an n x 1 numpy array where n is the number of examples in the file.
            The array must contain the Output column values of the file

    NOTE: Preserve the order of examples in the file
    """


    #importing output column from csv file as dataFrame
    dataFrame = pd.read_csv(file_path,usecols=['Output'])

    #converting to maatrix
    outputMatrix=dataFrame.values
    #returning the n*1 matrix
    return outputMatrix




def get_weight_vector(feature_matrix, output, lambda_reg, p):
    """
    feature_matrix: an n x m 2-D numpy array where n is the number of samples
                    and m the feature size.
    output: an n x 1 numpy array reprsenting the outputs for the n samples
    lambda_reg: regularization parameter
    p: p-norm for the regularized regression

    Return: an m x 1 numpy array weight vector obtained through stochastic gradient descent
            using the provided function parameters such that the matrix product
            of the feature_matrix matrix with this vector will give you the
            n x 1 regression outputs

    NOTE: While testing this function we will use feature_matrices not obtained
          from the get_feature_matrix() function but you can assume that all elements
          of this matrix will be of type float
    """

    alpha=0.00000001
    summ=0
    rows=feature_matrix.shape[0]
    columns=feature_matrix.shape[1 ]
    #lambda_reg=100
    weightVector=np.zeros((columns,1))
    #weightVector.fill(3)
    #print(weightVector)
    meanErr=0.0

   

    mean=np.mean(feature_matrix, axis=0)
    c_range = np.ptp(feature_matrix,axis=0)
    c_range = np.where(c_range==0, np.ones(np.shape(c_range)), c_range)
    #print (rows,columns)
    #scaling the features
    
    #for i in range(rows):
      #for j in range(columns):
        #feature_matrix[i,j]=((feature_matrix[i,j]-mean[j])/mean[j])
    #weightVector[1,0]=5.9009
    #print(weightVector[1,0])
    #print(feature_matrix[:,[1]], feature_matrix[[1]])
    #print(p*lambda_reg*(weightVector**(p-1)))
    for loop in range(10):
      for i in range(rows):
        #for j in range(columns):
          weightVector=weightVector   -   alpha*(   (2*((np.dot(feature_matrix[[i]],feature_matrix[[i]].T)[0,0])*weightVector)) - (2*feature_matrix[[i]]*output[i]).T  +  p*lambda_reg*(weightVector**(p-1))  )
          alpha=alpha/1.0001
          
    err=(np.mean((np.dot(feature_matrix,weightVector)-output)**2))**(1/2)
    print(err)
     
    return  weightVector
    


def get_my_best_weight_vector():
    """
    Return: your best m x 1 numpy array weight vector used to predict the output for the
            kaggle competition.

            The matrix product of the feature_matrix, obtained from get_feature_matrix()
            call with file as test_features.csv, with this weight vector should
            result in you best prediction for the test dataset.

    NOTE: For your final submission you are expected to provide an output.csv containing
          your predictions for the Kaggle test set and the weight vector returned here
          must help us to EXACTLY reproduce that output.csv

          We will also be using this weight to evaluate on a separate hold out test set

          We expect this function to return fast. So you are encouraged to return a pickeled
          file after all your experiments with various values of p and lambda_reg.
    """
    with open('weight.pickle', 'rb') as handle:
      res = pickle.load(handle)
    return res




def  main():
  trainMatrix=get_feature_matrix('train.csv')
  #print(trainMatrix)

  testFeaturesMatrix=get_feature_matrix('test_features.csv')
  #print(testFeaturesMatrix)
  #print("\n\n\n\n\n\n")
  outputVector=get_output('train.csv')
  #print(outputVector)
  weightVector=get_weight_vector(trainMatrix,outputVector,5,2)
  #print(trainMatrix[0])
  testWeightVector=np.dot(testFeaturesMatrix ,weightVector)
  #print(testWeightVector)
  #testDf=pd.DataFrame(data=testWeightVector[1:,0],  columns=testWeightVector[0,1:])
  #print(testDf)

  aa=pd.DataFrame(testWeightVector , columns=['Output'])
  #print(aa)
  aa.index +=1

  #testWeightVector.tofile('output.csv',sep=',',format='%10.5f')

  #to generate output csv file
  aa.to_csv('output.csv', index_label='Id')

  #saving the pickle file
  with open('weight.pickle', 'wb') as handle:
    pickle.dump(weightVector, handle, protocol=pickle.HIGHEST_PROTOCOL)
  best=get_my_best_weight_vector();
  





if __name__ == '__main__':
  main()