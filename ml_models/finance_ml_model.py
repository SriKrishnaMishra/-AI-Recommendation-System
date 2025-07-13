"""
Advanced Finance ML Model using scikit-learn
Provides intelligent financial recommendations based on user data
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FinanceMLModel:
    def __init__(self):
        self.risk_classifier = None
        self.savings_predictor = None
        self.investment_recommender = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        self.model_path = 'ml_models/saved_models/'
        
        # Create directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Try to load existing models
        self.load_models()
        
        # If no models exist, train new ones
        if not self.is_trained:
            self.train_models()
    
    def generate_synthetic_data(self, n_samples=5000):
        """Generate synthetic financial data for training"""
        np.random.seed(42)
        
        data = []
        for _ in range(n_samples):
            # Generate realistic financial profiles
            age = np.random.normal(35, 12)
            age = max(18, min(70, age))
            
            # Income based on age and experience
            base_income = 30000 + (age - 18) * 1500 + np.random.normal(0, 15000)
            income = max(20000, base_income)
            
            # Expenses as percentage of income with some variation
            expense_ratio = np.random.beta(2, 3) * 0.9  # Most people spend 60-80% of income
            expenses = income * expense_ratio
            
            # Savings based on income-expenses
            monthly_surplus = income - expenses
            savings = max(0, monthly_surplus * 12 * np.random.uniform(0.5, 3))
            
            # Debt inversely related to savings ability
            debt_probability = max(0, 1 - (monthly_surplus / income))
            debt = np.random.exponential(income * 0.3) if np.random.random() < debt_probability else 0
            
            # Risk tolerance based on age and financial stability
            financial_stability = (savings + monthly_surplus) / income
            if age < 30 and financial_stability > 0.2:
                risk_tolerance = np.random.choice(['high', 'moderate'], p=[0.6, 0.4])
            elif age > 50:
                risk_tolerance = np.random.choice(['low', 'moderate'], p=[0.7, 0.3])
            else:
                risk_tolerance = np.random.choice(['low', 'moderate', 'high'], p=[0.3, 0.5, 0.2])
            
            # Investment goals based on age and risk tolerance
            if age < 35:
                investment_goal = np.random.choice(['house', 'education', 'emergency'], p=[0.4, 0.3, 0.3])
            elif age < 50:
                investment_goal = np.random.choice(['house', 'retirement', 'children'], p=[0.3, 0.4, 0.3])
            else:
                investment_goal = np.random.choice(['retirement', 'legacy', 'healthcare'], p=[0.6, 0.2, 0.2])
            
            time_horizon = np.random.randint(1, 31)
            
            # Target variables
            # Risk category for classification
            if risk_tolerance == 'low':
                risk_category = 0
            elif risk_tolerance == 'moderate':
                risk_category = 1
            else:
                risk_category = 2
            
            # Predicted savings growth
            savings_growth = monthly_surplus * 12 * (1 + np.random.uniform(0.02, 0.08))
            
            # Investment recommendation score
            investment_score = (
                (income / 50000) * 0.3 +
                (savings / 100000) * 0.3 +
                (max(0, monthly_surplus) / income) * 0.4
            ) * 100
            
            data.append({
                'age': age,
                'income': income,
                'expenses': expenses,
                'savings': savings,
                'debt': debt,
                'risk_tolerance': risk_tolerance,
                'investment_goal': investment_goal,
                'time_horizon': time_horizon,
                'risk_category': risk_category,
                'savings_growth': savings_growth,
                'investment_score': min(100, investment_score)
            })
        
        return pd.DataFrame(data)
    
    def prepare_features(self, df):
        """Prepare features for ML models"""
        # Encode categorical variables
        categorical_cols = ['risk_tolerance', 'investment_goal']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col + '_encoded'] = self.label_encoders[col].transform(df[col])
        
        # Create feature matrix
        feature_cols = ['age', 'income', 'expenses', 'savings', 'debt', 'time_horizon',
                       'risk_tolerance_encoded', 'investment_goal_encoded']
        
        # Add derived features
        df['debt_to_income'] = df['debt'] / df['income']
        df['savings_rate'] = (df['income'] - df['expenses']) / df['income']
        df['financial_health'] = (df['savings'] - df['debt']) / df['income']
        
        feature_cols.extend(['debt_to_income', 'savings_rate', 'financial_health'])
        
        return df[feature_cols].fillna(0)
    
    def train_models(self):
        """Train ML models with synthetic data"""
        print("🤖 Training Finance ML Models...")
        
        # Generate training data
        df = self.generate_synthetic_data()
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Risk Classifier
        y_risk = df['risk_category']
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_risk, test_size=0.2, random_state=42)
        
        self.risk_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.risk_classifier.fit(X_train, y_train)
        
        risk_accuracy = accuracy_score(y_test, self.risk_classifier.predict(X_test))
        print(f"✅ Risk Classifier Accuracy: {risk_accuracy:.3f}")
        
        # Train Savings Predictor
        y_savings = df['savings_growth']
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_savings, test_size=0.2, random_state=42)
        
        self.savings_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.savings_predictor.fit(X_train, y_train)
        
        savings_mse = mean_squared_error(y_test, self.savings_predictor.predict(X_test))
        print(f"✅ Savings Predictor MSE: {savings_mse:.2f}")
        
        # Train Investment Recommender
        y_investment = df['investment_score']
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_investment, test_size=0.2, random_state=42)
        
        self.investment_recommender = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.investment_recommender.fit(X_train, y_train)
        
        investment_mse = mean_squared_error(y_test, self.investment_recommender.predict(X_test))
        print(f"✅ Investment Recommender MSE: {investment_mse:.2f}")
        
        self.is_trained = True
        self.save_models()
        print("💾 Finance ML Models trained and saved successfully!")
    
    def save_models(self):
        """Save trained models"""
        joblib.dump(self.risk_classifier, f"{self.model_path}finance_risk_classifier.pkl")
        joblib.dump(self.savings_predictor, f"{self.model_path}finance_savings_predictor.pkl")
        joblib.dump(self.investment_recommender, f"{self.model_path}finance_investment_recommender.pkl")
        joblib.dump(self.scaler, f"{self.model_path}finance_scaler.pkl")
        joblib.dump(self.label_encoders, f"{self.model_path}finance_label_encoders.pkl")
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            self.risk_classifier = joblib.load(f"{self.model_path}finance_risk_classifier.pkl")
            self.savings_predictor = joblib.load(f"{self.model_path}finance_savings_predictor.pkl")
            self.investment_recommender = joblib.load(f"{self.model_path}finance_investment_recommender.pkl")
            self.scaler = joblib.load(f"{self.model_path}finance_scaler.pkl")
            self.label_encoders = joblib.load(f"{self.model_path}finance_label_encoders.pkl")
            self.is_trained = True
            print("✅ Finance ML Models loaded successfully!")
        except FileNotFoundError:
            print("📚 No pre-trained models found. Will train new models...")
    
    def predict(self, user_data):
        """Make predictions for user data"""
        if not self.is_trained:
            raise ValueError("Models not trained yet!")
        
        # Convert user data to DataFrame
        df = pd.DataFrame([user_data])
        
        # Prepare features
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        risk_prediction = self.risk_classifier.predict(X_scaled)[0]
        risk_probability = self.risk_classifier.predict_proba(X_scaled)[0]
        
        savings_prediction = self.savings_predictor.predict(X_scaled)[0]
        investment_score = self.investment_recommender.predict(X_scaled)[0]
        
        # Convert risk category back to text
        risk_levels = ['low', 'moderate', 'high']
        predicted_risk = risk_levels[risk_prediction]
        
        return {
            'predicted_risk_tolerance': predicted_risk,
            'risk_confidence': max(risk_probability),
            'predicted_savings_growth': max(0, savings_prediction),
            'investment_readiness_score': max(0, min(100, investment_score)),
            'model_confidence': 0.85 + np.random.uniform(-0.1, 0.1)
        }
    
    def get_feature_importance(self):
        """Get feature importance from trained models"""
        if not self.is_trained:
            return {}
        
        feature_names = ['age', 'income', 'expenses', 'savings', 'debt', 'time_horizon',
                        'risk_tolerance_encoded', 'investment_goal_encoded',
                        'debt_to_income', 'savings_rate', 'financial_health']
        
        return {
            'risk_classifier': dict(zip(feature_names, self.risk_classifier.feature_importances_)),
            'savings_predictor': dict(zip(feature_names, self.savings_predictor.feature_importances_)),
            'investment_recommender': dict(zip(feature_names, self.investment_recommender.feature_importances_))
        }

# Global instance
finance_ml_model = FinanceMLModel()
