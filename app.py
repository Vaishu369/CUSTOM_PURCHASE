from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

app = Flask(__name__)

np.random.seed(42)
n_samples = 1000

age = np.random.randint(18, 66, n_samples) + np.random.normal(0, 2, n_samples)
income = np.random.randint(20000, 100001, n_samples) + np.random.normal(0, 2000, n_samples)

buy_prob = 1 / (1 + np.exp(-0.12*(age - 40) - 0.00012*(income - 50000)))
buy = (buy_prob > 0.5).astype(int)
flip_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
buy[flip_indices] = 1 - buy[flip_indices]

data = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Buy': buy
})

X = data[['Age', 'Income']]
y = data['Buy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('logreg', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
threshold = 0.5


@app.route('/')
def home():
    return redirect(url_for('predict'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    confidence = None
    message = None

    if request.method == 'POST':
        try:
            user_age = float(request.form['age'])
            user_income = float(request.form['income'])
            category = request.form.get('category', '')
            price_range = request.form.get('price_range', '')

            input_df = pd.DataFrame({'Age': [user_age], 'Income': [user_income]})
            pred_proba = pipeline.predict_proba(input_df)[0, 1]
            pred_binary = int(pred_proba > threshold)

            confidence = f"{pred_proba*100:.0f}%"

            if pred_binary == 1:
                prediction = "Yes"
                message = ("Great news! Based on your profile, this product is a perfect match for you. "
                           "You’re very likely to love and buy this item!")
            else:
                prediction = "No"
                message = ("This product might not be the best fit right now. Want to try another one? "
                           "Looks like this isn’t your ideal choice, but we have other recommendations!")

        except ValueError:
            prediction = "Invalid input."
            message = "Please enter valid numeric values for Age and Income."
        except Exception as e:
            prediction = "Error"
            message = f"Something went wrong: {str(e)}"

    return render_template('index.html', 
                           prediction=prediction, 
                           confidence=confidence,
                           message=message)


@app.route('/about')
def about():
    return '''
    <h2>How Our Purchase Prediction Works</h2>
    <p>Our AI model uses your age and income to predict how likely you are to purchase a product. 
    This prediction is based on patterns learned from thousands of users and their buying habits.</p>
    <p>We continuously improve the model to help you find the best matches for your lifestyle and budget.</p>
    <p><a href="/">Back to Prediction</a></p>
    '''


@app.route('/visualizations')
def visualizations():
    import matplotlib.pyplot as plt
    import io
    import base64
    import numpy as np
    from flask import render_template_string

    def plot_to_img():
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        return img

    plt.figure(figsize=(8,6))
    plt.scatter(X['Age'], X['Income'], c=y, cmap='bwr', alpha=0.6, edgecolor='k')
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title('All data: Age vs Income colored by Purchase')
    img1 = plot_to_img()

    age_min, age_max = X['Age'].min() - 2, X['Age'].max() + 2
    income_min, income_max = X['Income'].min() - 2000, X['Income'].max() + 2000

    age_grid, income_grid = np.meshgrid(
        np.linspace(age_min, age_max, 200),
        np.linspace(income_min, income_max, 200)
    )

    grid_points = pd.DataFrame({'Age': age_grid.ravel(), 'Income': income_grid.ravel()})
    probs = pipeline.predict_proba(grid_points)[:, 1].reshape(age_grid.shape)

    plt.figure(figsize=(8,6))
    contour = plt.contourf(age_grid, income_grid, probs, 25, cmap='RdBu', alpha=0.6)
    plt.colorbar(contour, label='Predicted Purchase Probability')
    plt.scatter(X['Age'], X['Income'], c=y, cmap='bwr', edgecolor='k', alpha=0.5)
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title('Predicted Purchase Probability and Data')
    img2 = plot_to_img()

    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    plt.figure(figsize=(7,5))
    plt.hist(y_pred_proba, bins=20, color='purple', alpha=0.7)
    plt.xlabel('Predicted Probability of Purchase')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Probabilities')
    img3 = plot_to_img()

    return render_template_string('''
    <h2>Visualizations</h2>
    <p><a href="{{ url_for('predict') }}">Back to Prediction</a></p>
    <h3>Data scatter plot</h3>
    <img src="data:image/png;base64,{{img1}}" alt="Scatter plot"><br><br>
    <h3>Probability surface and data</h3>
    <img src="data:image/png;base64,{{img2}}" alt="Probability surface"><br><br>
    <h3>Histogram of predicted probabilities</h3>
    <img src="data:image/png;base64,{{img3}}" alt="Probability histogram">
    ''', img1=img1, img2=img2, img3=img3)


if __name__ == '__main__':
    app.run(debug=True)
