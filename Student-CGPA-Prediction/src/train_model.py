import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_model(df):
    X = df.drop('Current_CGPA', axis=1)
    y = df['Current_CGPA']
    print("Training features:", X.columns.tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, 'model/cgpa_prediction_model.pkl')

    return model, X_test, y_test