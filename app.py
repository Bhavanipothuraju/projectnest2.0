from flask import Flask, render_template, request, redirect, url_for, flash
from model import train_model
import pandas as pd

app = Flask(__name__)
app.secret_key = 'secretivekey'

# Train the model and get the imputer
model, imputer, feature_names = train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Add your login logic here
        username = request.form['username']
        password = request.form['password']

        # Example: Check if the login credentials are valid (replace with your actual login logic)
        if username == 'baby' and password == 'rani':
            # Redirect to the predict page only if the login is successful
            flash('Login successful', 'success')  # Optional: Use flash for displaying messages
            return redirect(url_for('predict'))  # Use url_for to ensure proper routing

        else:
            # Optionally, provide a message for unsuccessful login
            flash('Invalid username or password', 'error')

    # Render the login page if the request method is GET or if the login is unsuccessful
    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        transaction_id = request.form['transaction_id']
        amount = request.form['amount']

        # Create a DataFrame with user input features
        user_input = pd.DataFrame({'id': [float(transaction_id)], 'Amount': [float(amount)]})

        # Use only 'id' and 'Amount' columns for prediction
        user_input = user_input[['id', 'Amount']]

        # Impute missing values in user_input using mean imputation
        user_input = imputer.transform(user_input)

        # Make a prediction
        prediction = model.predict(user_input)

        # Prepare the result message
        result_message = "The transaction is predicted as fraudulent." if prediction[0] == 1 else "The transaction is predicted as non-fraudulent."

        # Pass the prediction value and other details as query parameters
        return redirect(url_for('result', prediction=prediction[0], result_message=result_message, transaction_id=transaction_id, amount=amount))

    return render_template('predict.html')

@app.route('/result')
def result():
    # Check if prediction and result_message are present in the query parameters
    prediction = request.args.get('prediction', '')
    result_message = request.args.get('result_message', '')

    if not prediction or not result_message:
        # Redirect to the predict page if the necessary information is not present
        flash('Invalid access to the result page', 'error')
        return redirect(url_for('predict'))

    # Get additional information (transaction_id and amount) from the query parameters
    transaction_id = request.args.get('transaction_id', '')
    amount = request.args.get('amount', '')

    return render_template('result.html', prediction=prediction, result_message=result_message, transaction_id=transaction_id, amount=amount)

@app.route('/contact')
def contact():
    # Add your contact page logic here
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
