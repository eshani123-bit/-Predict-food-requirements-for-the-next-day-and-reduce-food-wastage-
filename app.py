from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

menu_map = {'Normal': 0, 'Special': 1}
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        people = int(request.form['people'])
        menu = menu_map[request.form['menu']]
        event = 1 if request.form['event'] == 'Yes' else 0
        day = day_map[request.form['day']]

        X_input = np.array([[people, menu, event, day]])
        pred = model.predict(X_input)[0]
        recommended = round(pred * 1.05, 2)  # Add 5% margin
        prediction = f"Recommend preparing approximately {recommended} Kg of food."
        
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)