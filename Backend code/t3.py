import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # Handles imbalanced datasets

class ChurnPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.model = None
        self.target_col = 'Churn'
        self.feature_columns = None

    def _preprocess_data(self, df, training=True):
        """Universal preprocessing pipeline"""
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])

        if self.target_col in df.columns:
            # Ensure Churn is binary (0 or 1)
            df[self.target_col] = df[self.target_col].apply(lambda x: 1 if x > 0 else 0)
            
            # Confirm the Churn column is binary
            if len(df[self.target_col].unique()) != 2:
                raise ValueError("Churn column must be binary (0/1 or Yes/No)")

        categorical_cols = df.select_dtypes(exclude=np.number).columns
        if self.target_col in categorical_cols:
            categorical_cols = categorical_cols.drop(self.target_col)
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        if training:
            self.feature_columns = df.drop(columns=[self.target_col]).columns.tolist()
        elif self.feature_columns:
            df = df.reindex(columns=self.feature_columns, fill_value=0)

        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df

    def _train_model(self, X_train, y_train):
        """Train a new model with balanced data"""
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train_resampled, y_train_resampled)

        joblib.dump(self.model, 'churn_model.pkl')
        joblib.dump(self.scaler, 'churn_scaler.pkl')
        joblib.dump(self.feature_columns, 'feature_columns.pkl')

    def analyze_dataset(self, file_path):
        """Main function: Determines whether to train or predict"""
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            processed_df = self._preprocess_data(df, training=self.target_col in df.columns)

            if self.target_col in processed_df.columns:
                X = processed_df.drop(columns=[self.target_col])
                y = processed_df[self.target_col]

                # Check if the target is continuous, and if so, convert to binary
                if not y.isin([0, 1]).all():
                    y = y.apply(lambda x: 1 if x > 0 else 0)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                self._train_model(X_train, y_train)
                churn_rate = y.mean() * 100
                print(f"\n✅ Training Complete! Churn Rate: {churn_rate:.2f}%")
                return churn_rate
            else:
                self.model = joblib.load('churn_model.pkl')
                self.scaler = joblib.load('churn_scaler.pkl')
                self.feature_columns = joblib.load('feature_columns.pkl')

                processed_df = processed_df.reindex(columns=self.feature_columns, fill_value=0)
                predictions = self.model.predict(processed_df)
                churn_rate = predictions.mean() * 100
                print(f"\n🔍 Predicted Churn Rate: {churn_rate:.2f}%")
                return churn_rate

        except FileNotFoundError:
            print("❌ No trained model found. Train the model first with a dataset that contains 'Churn'.")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            return None

# Usage
if __name__ == "__main__":
    predictor = ChurnPredictor()
    
    # First run: Provide dataset WITH "Churn" column to train model
    # predictor.analyze_dataset("train_data.csv")
    
    # Subsequent runs: Use dataset WITHOUT "Churn" for predictions
    predictor.analyze_dataset("food_delivery_service_data.csv")
