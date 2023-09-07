# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



#This creates the path to call my EEG data
#
#IMPORTANT!!!!
#You will have to change the file path depending on where the Blink_data.csv is stored
#

D = r'C:\Users\asuka\OneDrive\Desktop\Emo\Blink_data.csv'

#Complete data set


Data = pd.read_csv(D)


yD = Data.Blink_Averaged
featuresD = ['0_Der_avg','1_Der_avg','3_Der_avg']
XD = Data[featuresD]
#This assigns the EEG chanels 0,1 & 3 as the input values/x value



train_XD, val_XD, train_yD, val_yD = train_test_split(XD,yD, random_state=2)



EEG_model1 = DecisionTreeClassifier(random_state=10)

EEG_model1.fit(train_XD,train_yD)

def pred(data):
    output = EEG_model1.predict(data)
    
    return output
    

