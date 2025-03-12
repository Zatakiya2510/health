import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# Train and test models
accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracies[name] = round(accuracy_score(y_test, predictions) * 100, 2)

# Load models
try:
    diabetes_model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    print(f"Error loading models: {e}")

app = Flask(__name__, template_folder='template')
app.secret_key = 'your_secret_key'

# Hardcoded user credentials and dynamically stored users
users = {'admin': 'password123', 'user': 'userpass'}

@app.route('/')
def home():
    if 'user' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))
    return render_template('webpage.html')

@app.route('/goback')
def goback():
    return redirect(url_for('home'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash('Username already exists. Choose another.', 'danger')
        else:
            users[username] = password
            flash('Signup successful! You can now log in.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['user'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid user, please sign up first.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if 'user' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))
    if request.method == 'POST':
        try:
            numerical_features = [
                'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                'insulin', 'bmi', 'diabetes_pedigree_function', 'age'
            ]
            input_features = [float(request.form.get(col, 0)) for col in numerical_features]

            input_array = np.array([input_features]).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            output = diabetes_model.predict(input_scaled)

            if output[0] == 0:
                result = "You Do Not Have Diabetes 😊"
                medicine = "No medication required. Maintain a healthy lifestyle with a balanced diet and regular exercise."
                precautions = "Eat a balanced diet, exercise regularly, and maintain a healthy weight."
                diet = "Reduce sugar intake, eat whole grains, lean protein, and healthy fats. Stay hydrated."
                sugar_consumption = "Limit sugar consumption to under 25g per day."
            else:
                result = "It Seems You Have Diabetes 😞"
                medicine = "Recommended Medicine: Metformin (for Type 2 Diabetes). Please consult a doctor for proper medication and dosage."
                precautions = "Monitor blood sugar levels, maintain a healthy diet, exercise regularly, and avoid stress."
                diet = "Follow a low-carb, high-fiber diet. Eat fruits, vegetables, whole grains, and avoid processed sugar."
                sugar_consumption = "Limit sugar consumption strictly and focus on natural sugars from fruits in moderation."

            return render_template('diabetes_form.html', data=result, medicine=medicine, precautions=precautions, diet=diet, sugar_consumption=sugar_consumption, accuracies=accuracies)

        except Exception as e:
            print(f"Error in Diabetes Prediction: {e}")
            return render_template('diabetes_form.html', data="Error in processing input", medicine="", precautions="", diet="", sugar_consumption="", accuracies=accuracies)

    return render_template('diabetes_form.html', accuracies=accuracies)

@app.route('/accuracy')
def accuracy():
    if 'user' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))
    return render_template('accuracy.html', accuracies=accuracies)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
