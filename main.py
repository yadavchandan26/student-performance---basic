#this is the main file . the most important file which carries unlimited aura 
#aura+++

from performance_prediction.data_preprocessing import processing
from performance_prediction.model_training import train_model
from performance_prediction.evaluation import evaluate_model , generate_submission

Feature_cols=['English_Score','Math_Score','Science_Score']
Data_path="data/student_performance.csv"

def main():
    x_train,x_test,y_train,y_test,id_test=processing(
        Data_path,Feature_cols
    )
    model=train_model(x_train,y_train)

    evaluate_model(model,x_train,y_train)
    y_pred=model.predict(x_test)
    generate_submission(id_test,y_pred,output_path="submission.csv")

if __name__=="__main__":
    main()
