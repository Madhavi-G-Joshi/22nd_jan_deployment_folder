from flask import Flask

#initialize the app

app=Flask(__name__)


@app.route('/')
def home():
    return 'hello world'

@app.route('/madhavi')
def madhavi():
    return 'hello madhavi'

#run the app
app.run()