# Object-Characteristics-Tracker

Incorporting VGG-Face and YoloV4 I was able to create an flask application that can toggle between different models that can predict age, race, gender, and emotion all while tracking an object. 

# How to run:

1) Git clone project
2) Go to project directory
3) Setup env (optional, but recommended) ->  

`
pip install -r requirements.txt
`

4) Weights and Pretrained Models (Google Drive since they are too large for github):

Weights (Add to data folder): https://drive.google.com/file/d/1v1GZ0KOAw7I0Tnks5hLRI4m7oxCja1xn/view?usp=sharing
Pretrained Models (download with checkpoint folder): https://drive.google.com/drive/folders/130YXUr-WS3LpaZFC80spgZTsFi9rJ6z3?usp=sharing

5) Flask Setup and Run: 

`
export FLASK_APP=object_tracker
flask run
`

# Output
Screenshot:

https://raw.githubusercontent.com/Aschonn/Object-Characteristics-Tracker/main/output.png
