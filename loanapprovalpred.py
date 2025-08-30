import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_score, recall_score, f1_score, 
                           roc_auc_score, accuracy_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class LoanApprovalPredictor:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        
    def load_data(self, file_path):
        """Load and inspect the dataset"""
        try:
            self.data = pd.read_csv(file_path)
            
            # FIX: Remove leading/trailing spaces from column names
            self.data.columns = self.data.columns.str.strip()
            
            print(f"Dataset loaded successfully with shape: {self.data.shape}")
            print("\nDataset Info:")
            print(self.data.info())
            print("\nFirst 5 rows:")
            print(self.data.head())
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def explore_data(self):
        """Explore the dataset and check for class imbalance"""
        print("\n=== DATA EXPLORATION ===")
        
        # Check for missing values
        print("\nMissing values per column:")
        print(self.data.isnull().sum())
        
        # Check class distribution
        print("\nLoan Status Distribution:")
        loan_status_counts = self.data['loan_status'].value_counts()
        print(loan_status_counts)
        
        # Visualize class distribution
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        loan_status_counts.plot(kind='bar', color=['skyblue', 'lightcoral'])
        plt.title('Loan Status Distribution')
        plt.xlabel('Loan Status')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        # Check data types
        print("\nData Types:")
        print(self.data.dtypes)
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(self.data.describe())
        
        plt.tight_layout()
        plt.show()
        
        return loan_status_counts
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("\n=== HANDLING MISSING VALUES ===")
        
        # Create a copy to avoid modifying original data
        data_clean = self.data.copy()
        
        # Identify numerical and categorical columns
        numerical_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data_clean.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable from numerical columns if present
        if 'loan_id' in numerical_cols:
            numerical_cols.remove('loan_id')
        if 'loan_status' in numerical_cols:
            numerical_cols.remove('loan_status')
        
        # Handle numerical missing values with median
        for col in numerical_cols:
            if data_clean[col].isnull().sum() > 0:
                median_val = data_clean[col].median()
                data_clean[col].fillna(median_val, inplace=True)
                print(f"Filled missing values in {col} with median: {median_val}")
        
        # Handle categorical missing values with mode
        for col in categorical_cols:
            if col != 'loan_status' and data_clean[col].isnull().sum() > 0:
                mode_val = data_clean[col].mode()[0]
                data_clean[col].fillna(mode_val, inplace=True)
                print(f"Filled missing values in {col} with mode: {mode_val}")
        
        print("Missing values handling completed.")
        self.data = data_clean
        return self.data
    
    def encode_categorical_features(self):
        """Encode categorical variables"""
        print("\n=== ENCODING CATEGORICAL FEATURES ===")
        
        data_encoded = self.data.copy()
        
        # Identify categorical columns (excluding target if it's categorical)
        categorical_cols = data_encoded.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable if it's in categorical columns
        if 'loan_status' in categorical_cols:
            categorical_cols.remove('loan_status')
        
        # Use Label Encoding for categorical variables
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
            label_encoders[col] = le
            print(f"Encoded column: {col}")
        
        # Encode target variable if it's categorical
        if data_encoded['loan_status'].dtype == 'object':
            le_target = LabelEncoder()
            data_encoded['loan_status'] = le_target.fit_transform(data_encoded['loan_status'])
            print("Encoded target variable: loan_status")
            # Map for interpretation: 0 = Rejected, 1 = Approved
            print("Target mapping: 0 = Rejected, 1 = Approved")
        
        self.data = data_encoded
        return self.data, label_encoders
    
    def prepare_data(self, test_size=0.2):
        """Prepare data for modeling"""
        print("\n=== PREPARING DATA FOR MODELING ===")
        
        # Remove loan_id as it's not a feature
        if 'loan_id' in self.data.columns:
            self.data = self.data.drop('loan_id', axis=1)
        
        # Separate features and target
        X = self.data.drop('loan_status', axis=1)
        y = self.data['loan_status']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")
        print(f"Class distribution in training set: {np.bincount(self.y_train)}")
        print(f"Class distribution in testing set: {np.bincount(self.y_test)}")
    
    def handle_imbalance_smote(self):
        """Apply SMOTE to handle class imbalance"""
        print("\n=== APPLYING SMOTE FOR CLASS IMBALANCE ===")
        
        smote = SMOTE(random_state=42)
        self.X_train_smote, self.y_train_smote = smote.fit_resample(
            self.X_train_scaled, self.y_train
        )
        
        print(f"After SMOTE - Training set shape: {self.X_train_smote.shape}")
        print(f"Class distribution after SMOTE: {np.bincount(self.y_train_smote)}")
        
        return self.X_train_smote, self.y_train_smote
    
    def train_models(self, use_smote=True):
        """Train multiple models and compare performance"""
        print("\n=== TRAINING MODELS ===")
        
        if use_smote:
            X_train = self.X_train_smote
            y_train = self.y_train_smote
        else:
            X_train = self.X_train_scaled
            y_train = self.y_train
        
        # Define models to train
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            if y_pred_proba is not None:
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            else:
                roc_auc = None
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            if roc_auc is not None:
                print(f"ROC-AUC: {roc_auc:.4f}")
            
            # Print classification report
            print(f"\nClassification Report for {name}:")
            print(classification_report(self.y_test, y_pred))
        
        self.models = results
        return results
    
    def find_best_model(self):
        """Find the best model based on F1-score"""
        best_model_name = None
        best_f1 = 0
        
        for name, result in self.models.items():
            if result['f1_score'] > best_f1:
                best_f1 = result['f1_score']
                best_model_name = name
        
        if best_model_name:
            self.best_model = self.models[best_model_name]['model']
            print(f"\nBest Model: {best_model_name} with F1-Score: {best_f1:.4f}")
        
        return best_model_name, self.best_model
    
    def evaluate_best_model(self):
        """Comprehensive evaluation of the best model"""
        if self.best_model is None:
            print("No best model found. Please train models first.")
            return
        
        best_model_name = self.find_best_model()[0]
        result = self.models[best_model_name]
        
        print(f"\n=== COMPREHENSIVE EVALUATION OF {best_model_name} ===")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, result['predictions'])
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Rejected', 'Approved'],
                   yticklabels=['Rejected', 'Approved'])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Feature Importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            plt.subplot(1, 2, 2)
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.best_model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title('Top 10 Feature Importance')
        
        plt.tight_layout()
        plt.show()
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.best_model, 
            self.X_train_scaled, 
            self.y_train, 
            cv=5, 
            scoring='f1'
        )
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def run_complete_analysis(self, file_path):
        """Run the complete analysis pipeline"""
        print("Starting Loan Approval Prediction Analysis...")
        
        # Step 1: Load data
        if not self.load_data(file_path):
            return
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Handle missing values
        self.handle_missing_values()
        
        # Step 4: Encode categorical features
        self.encode_categorical_features()
        
        # Step 5: Prepare data for modeling
        self.prepare_data()
        
        # Step 6: Handle class imbalance with SMOTE
        self.handle_imbalance_smote()
        
        # Step 7: Train models
        self.train_models(use_smote=True)
        
        # Step 8: Find and evaluate best model
        self.find_best_model()
        self.evaluate_best_model()
        
        print("\n=== ANALYSIS COMPLETED ===")

# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = LoanApprovalPredictor()
    
    # Set your specific file path
    file_path = "loan_approval_dataset.csv"  # Update this to your file path
    
    try:
        predictor.run_complete_analysis(file_path)
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please make sure the file path is correct and the dataset is properly formatted.")