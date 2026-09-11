import pandas as pd


def evaluate_model(model, x_train, y_train):
    
    score = model.score(x_train, y_train)
    print(f"Training accuracy: {score:.4f}")
    return score


def generate_submission(id_test, y_pred, output_path='submission.csv'):
    
    submission = pd.DataFrame({
        'Student_ID': id_test,
        'Grade': y_pred
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    return submission
