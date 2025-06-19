from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model, evaluate_model

def main():
    df = load_data('data/transactions.csv')
    df = preprocess_data(df)

    # Train model
    model, X_test, y_test = train_model(df)

    # Evaluate
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
