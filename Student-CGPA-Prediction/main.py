from src.preprocess import preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model

def main():
    df = preprocess_data('data/student_data.csv')
    model, X_test, y_test = train_model(df)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()