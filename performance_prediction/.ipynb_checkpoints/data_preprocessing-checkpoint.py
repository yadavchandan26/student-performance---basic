"""this is the file named data_preprocessing where we will
    bascially load the data , perform some feature engineering 
    & some other stuff"""

import pandas as pd
from sklearn.model_selection import train_test_split

#performing loading dataset part
def load_data(data_path):
    data=pd.read_csv(data_path)
    return data


def total_marks(data):
    data['Total_Score']=data['Math_Score']+data['Science_Score']+data['English_Score']
    return data

def percentage_calc(data):
    data['Percentage']=(data['Total_Score']/3)
    return data
    
#performaing grade distribution part
def grade_provider(Percentage):
    if Percentage > 90 :
        return 'A'
    elif Percentage <=90 and Percentage >80:
        return 'B'
    elif Percentage <=80 and Percentage >70:
        return 'C'
    elif Percentage <=70 and Percentage >36:
        return 'Pass'
    else:
        return 'F'

def add_grades(data):
    data['Grade']=data['Percentage'].apply(grade_provider)
    return data
    
def features_selection(data):
    feature_cols=['English_Score','Math_Score','Science_Score']
    x=data[feature_cols]
    y=data['Grade']
    return x,y
    
def processing(data_path,feature_cols):
    data=load_data(data_path)
    
    data=total_marks(data)
    data=percentage_calc(data)
    data=add_grades(data)

    x,y=features_selection(data)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=42)

    return x_train,x_test,y_train,y_test
    

