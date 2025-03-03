from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Personal API"

@app.route('/name')
def name():
    return "Name: vishnuraju v"

@app.route('/register_number')
def register_number():
    return "Register Number: 22IT056"

@app.route('/department')
def department():
    return "Department: Information Technology"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  