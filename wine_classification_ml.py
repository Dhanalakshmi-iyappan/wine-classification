
"""
Wine Classification - Complete ML Pipeline
Algorithms: LightGBM, XGBoost, AdaBoost, CatBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load and clean Wine Classification dataset"""
    print("🍷 Loading Wine Classification Dataset...")

    # Create sample wine data
    np.random.seed(42)
    n_samples = 178

    # Generate realistic wine data
    data = {
        'alcohol': np.random.normal(13.0, 1.0, n_samples),
        'malic_acid': np.random.normal(2.3, 1.1, n_samples),
        'ash': np.random.normal(2.4, 0.3, n_samples),
        'alcalinity': np.random.normal(19.5, 3.3, n_samples),
        'magnesium': np.random.normal(99.7, 14.3, n_samples),
        'phenols': np.random.normal(2.3, 0.6, n_samples),
        'flavanoids': np.random.normal(2.0, 1.0, n_samples),
        'nonflavanoids': np.random.normal(0.36, 0.12, n_samples),
        'proanthocyanins': np.random.normal(1.6, 0.57, n_samples),
        'color_intensity': np.random.normal(5.1, 2.3, n_samples),
        'hue': np.random.normal(0.96, 0.23, n_samples),
        'diluted': np.random.normal(2.6, 0.71, n_samples),
        'proline': np.random.normal(746, 315, n_samples),
    }

    # Create wine classes (0, 1, 2)
    data['wine_class'] = np.repeat([0, 1, 2], [59, 71, 48])

    df = pd.DataFrame(data)

    print(f"📊 Original shape: {df.shape}")
    print(f"📊 Features: {len(df.columns)-1}")

    # Data Cleaning
    print("\n🧹 Starting Data Cleaning...")

    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values: {missing_values.sum()}")

    # Remove any missing values
    df = df.dropna()

    # Check for outliers and handle them
    feature_cols = [col for col in df.columns if col != 'wine_class']

    # Remove extreme outliers (beyond 3 standard deviations)
    for col in feature_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df = df[(df[col] >= mean_val - 3*std_val) & (df[col] <= mean_val + 3*std_val)]

    # Features and target
    X = df[feature_cols]
    y = df['wine_class'].astype(int)

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    print(f"✅ Cleaned shape: {X_scaled.shape}")
    print(f"✅ Classes distribution: {np.bincount(y)}")

    return X_scaled, y, scaler

def train_models(X, y):
    """Train all boosting models"""
    print("\n🚀 Training Models...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    # 1. LightGBM
    print("\n📊 Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        random_state=42,
        verbose=-1,
        n_estimators=100
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    lgb_accuracy = accuracy_score(y_test, lgb_pred)
    results['LightGBM'] = {'model': lgb_model, 'accuracy': lgb_accuracy, 'predictions': lgb_pred}

    # 2. XGBoost
    print("📊 Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss',
        n_estimators=100
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    results['XGBoost'] = {'model': xgb_model, 'accuracy': xgb_accuracy, 'predictions': xgb_pred}

    # 3. AdaBoost
    print("📊 Training AdaBoost...")
    ada_model = AdaBoostClassifier(
        random_state=42, 
        algorithm='SAMME',
        n_estimators=100
    )
    ada_model.fit(X_train, y_train)
    ada_pred = ada_model.predict(X_test)
    ada_accuracy = accuracy_score(y_test, ada_pred)
    results['AdaBoost'] = {'model': ada_model, 'accuracy': ada_accuracy, 'predictions': ada_pred}

    # 4. CatBoost
    print("📊 Training CatBoost...")
    cat_model = cb.CatBoostClassifier(
        iterations=100,
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict(X_test)
    cat_accuracy = accuracy_score(y_test, cat_pred)
    results['CatBoost'] = {'model': cat_model, 'accuracy': cat_accuracy, 'predictions': cat_pred}

    return results, X_test, y_test

def feature_importance_analysis(results):
    """Analyze feature importance across models"""
    print("\n📊 Feature Importance Analysis:")
    print("-" * 40)

    for name, result in results.items():
        model = result['model']
        print(f"\n{name} Top 5 Features:")

        if hasattr(model, 'feature_importances_'):
            # Get feature names (assuming we know the order)
            feature_names = ['alcohol', 'malic_acid', 'ash', 'alcalinity', 'magnesium', 
                           'phenols', 'flavanoids', 'nonflavanoids', 'proanthocyanins', 
                           'color_intensity', 'hue', 'diluted', 'proline']

            importances = model.feature_importances_
            feature_imp = list(zip(feature_names, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)

            for feat, imp in feature_imp[:5]:
                print(f"  {feat}: {imp:.4f}")

def evaluate_models(results, X_test, y_test):
    """Evaluate and compare all models"""
    print("\n📈 Model Evaluation Results:")
    print("=" * 50)

    for name, result in results.items():
        accuracy = result['accuracy']
        print(f"{name}: {accuracy:.4f}")

    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']

    print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

    # Detailed evaluation of best model
    best_pred = results[best_model_name]['predictions']
    print(f"\n📊 Detailed Results for {best_model_name}:")
    print("Classification Report:")
    print(classification_report(y_test, best_pred, 
                              target_names=['Class 0', 'Class 1', 'Class 2']))

def main():
    """Main execution function"""
    print("🍷 WINE CLASSIFICATION PIPELINE")
    print("=" * 50)

    # Load and clean data
    X, y, scaler = load_and_clean_data()

    # Train models
    results, X_test, y_test = train_models(X, y)

    # Evaluate models
    evaluate_models(results, X_test, y_test)

    # Feature importance analysis
    feature_importance_analysis(results)

    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
