import sys
import pandas as pd
import numpy as np
from DecisionTree import DecisionTree
from DecisionTree import CalculateAccuracy
from PruneFunction import PruneByClassificationError

num = "1"
train =   "dataset_"+num+"/training.csv"
val =     "dataset_"+num+"/validation.csv"
test =    "dataset_"+num+"/testing.csv"
toPrune = "y"
toPrint = "y"
methodToUse = "Entropy" # or "VarianceImpurity"

#Getting all columns and rows from the file
train = pd.read_csv(train)
val = pd.read_csv(val)
test = pd.read_csv(test)

train_X = train.iloc[ : , 0: train.shape[1]  - 1  ].values
train_Y = train.iloc[ : , train.shape[1]  - 1: train.shape[1] ].values

val_X = val.iloc[ : , 0: val.shape[1] - 1  ].values
val_Y = val.iloc[ : , val.shape[1] - 1: val.shape[1] ].values

test_X = test.iloc[ : , 0: test.shape[1] - 1  ].values
test_Y = test.iloc[ : , test.shape[1] - 1: test.shape[1] ].values

dt = DecisionTree( train_X, train_Y, methodToUse )
print( "\nMethod of " + methodToUse + " used to built DecisionTree:- ")

print( "Accuracy before Pruning on Validation Data :- " + str(CalculateAccuracy( dt, val_X, val_Y  ) ) )
print( "Accuracy before Pruning on Testing Data    :- " + str(CalculateAccuracy( dt, test_X, test_Y  ) ) )

if toPrune.lower() == "y" :
  dt = PruneByClassificationError( dt, np.concatenate((val_X,val_Y),axis=1) )
  print( "\nAccuracy after Pruning on Validation Data :- " + str(CalculateAccuracy( dt, val_X, val_Y  ) ) )
  print( "Accuracy after Pruning on Testing Data    :- " + str(CalculateAccuracy( dt, test_X, test_Y  ) ) )

if toPrint.lower() == "y":
  print ("Tree :- ")
  dt.PrintTree( train.columns )
