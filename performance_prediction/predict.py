import joblib 

def predict_grade(math_score,science_score,english_score):
    model=joblib.load('saved_model.pkl')

    features = [[english_score, math_score, science_score]]
    
    prediction = model.predict(features)
    return prediction[0]

if __name__ == "__main__":
    math = float(input("Math score: "))
    science = float(input("Science score: "))
    english = float(input("English score: "))
    
    grade = predict_grade(math, science, english)
    print(f"Predicted Grade: {grade}")