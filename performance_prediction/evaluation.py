import pandas as pd


def evaluate_model(model, x_train, y_train):
    
    score = model.score(x_train, y_train)
    print(f"Training accuracy: {score:.4f}")
    return score


def generate_submission(data, y_pred, output_path='submission.csv'):
    
    submission = pd.DataFrame({
        'Student_ID	': data['Student_ID	'],
        'Grade': y_pred
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    return submission
