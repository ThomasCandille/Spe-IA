# Import Required Libraries
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def preprocess_data(df, keep_passenger_id=False):
    sex_mapping = {'male': 1, 'female': 2}
    embarked_mapping = {'S': 1, 'C': 2, 'Q': 3}

    # Clean data
    df = df.drop_duplicates()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Drop useless features
    cols_to_drop = ['Name', 'Ticket', 'Cabin']
    if not keep_passenger_id:
        cols_to_drop.append('PassengerId')
    df = df.drop(columns=cols_to_drop)

    # Map categorical to numeric
    df['Sex'] = df['Sex'].map(sex_mapping)
    df['Embarked'] = df['Embarked'].map(embarked_mapping)

    # Create Age groups
    df['Age_Group'] = pd.cut(df['Age'],
                              bins=[0, 12, 20, 30, 40, 50, 60, 70, 80],
                              labels=[1, 2, 3, 4, 5, 6, 7, 8])
    df['Age_Group'] = df['Age_Group'].astype(int)

    return df

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

    return model

def main():
    # Get script directory
    script_dir = Path(__file__).parent

    # Load and preprocess training data
    df = pd.read_csv(script_dir / 'train.csv')
    print(f"Original shape: {df.shape}, Duplicates: {df.duplicated().sum()}")
    df = preprocess_data(df)
    print(f"After preprocessing: {df.shape}")
    print(df.head())

    # Split data
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")

    # Train model
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    # Process test.csv and make predictions
    df_test = pd.read_csv(script_dir / 'test.csv')
    passenger_ids = df_test['PassengerId'].copy()
    df_test = preprocess_data(df_test)
    predictions = model.predict(df_test)
    print(f"\nPredictions - Not Survived: {sum(predictions == 0)}, Survived: {sum(predictions == 1)}")

    # Create submission file
    submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
    submission.to_csv(script_dir / 'submission.csv', index=False)
    print("\nSubmission file created: submission.csv")
    print(submission.head(20))

if __name__ == "__main__":
    main()