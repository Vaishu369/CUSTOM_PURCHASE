import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

np.random.seed(42)
n_samples = 1000

age = np.random.randint(18, 66, n_samples) + np.random.normal(0, 2, n_samples)
income = np.random.randint(20000, 100001, n_samples) + np.random.normal(0, 2000, n_samples)

buy_prob = 1 / (1 + np.exp(-0.12*(age - 40) - 0.00012*(income - 50000)))
buy = (buy_prob > 0.5).astype(int)
flip_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
buy[flip_indices] = 1 - buy[flip_indices]

data = pd.DataFrame({'Age': age, 'Income': income, 'Buy': buy})

X = data[['Age', 'Income']]
y = data['Buy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('logreg', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'logistic_regression_poly_model.joblib')

threshold = 0.5

y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > threshold).astype(int)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")

age_min, age_max = X['Age'].min() - 2, X['Age'].max() + 2
income_min, income_max = X['Income'].min() - 2000, X['Income'].max() + 2000

age_grid, income_grid = np.meshgrid(
    np.linspace(age_min, age_max, 200),
    np.linspace(income_min, income_max, 200)
)

grid_points = pd.DataFrame({'Age': age_grid.ravel(), 'Income': income_grid.ravel()})
probs = pipeline.predict_proba(grid_points)[:, 1].reshape(age_grid.shape)

plt.figure(figsize=(9,7))
contour = plt.contourf(age_grid, income_grid, probs, 25, cmap='RdBu', alpha=0.7)
plt.colorbar(contour, label='Predicted Purchase Probability')
plt.scatter(X_train['Age'], X_train['Income'], c=y_train, cmap='bwr', edgecolor='k', alpha=0.5, label='Train data')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Predicted Purchase Probability and Training Data')
plt.legend()
plt.show()

def predict_user_input():
    try:
        user_age = float(input("Enter Age: "))
        user_income = float(input("Enter Income: "))

        input_df = pd.DataFrame({'Age': [user_age], 'Income': [user_income]})
        pred_proba = pipeline.predict_proba(input_df)[0, 1]
        pred_binary = int(pred_proba > threshold)

        print(f"\nPrediction for Age={user_age}, Income={user_income}:")
        print(f"  Purchase: {'Yes' if pred_binary == 1 else 'No'}")
        print(f"  Confidence (probability): {pred_proba:.3f}")
    except ValueError:
        print("Invalid input. Please enter numeric values for Age and Income.")

predict_user_input()
