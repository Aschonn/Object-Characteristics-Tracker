# basic imports

import os
import time
import tensorflow as tf
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# deep sort imports

from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

#deepface imports needed

from deepface import DeepFace
from deepface.commons import functions
from deepface.detectors import FaceDetector
from deepface.extendedmodels import Age

# flask imports 

from flask import Flask, render_template, request, Response
app = Flask(__name__)


# Starts Video

try:
    camera = cv2.VideoCapture(cv2.CAP_V4L2) 
except:
    camera = cv2.VideoCapture(0)


# Determines if either model is used

detection = 0
race = 0
gender = 0
age = 0
emotion = 0
video = 0
max_cosine_distance = 0.4
nn_budget = None
model_filename = 'model_data/mars-small128.pb'
weights = './checkpoints/yolov4-416'

#Encoder

encoder = gdet.create_box_encoder(model_filename, batch_size=1)

# Initalize Tracker

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Loads Yolo Model

saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']



# Initalizing Models

print("initalizing Face Detector")
face_detector = FaceDetector.build_model('opencv')
print("face detector has been completed")
print("initalizing Age Model")
age_model = DeepFace.build_model('Age')
print("finished Age Model")
print("initalizing Emotion Model")
emotion_model = DeepFace.build_model('Emotion')
print("finished Emotion Model")
print("initalizing Gender Model")
gender_model = DeepFace.build_model('Gender')
print("finished Gender Model")
print("initalizing Race Model")
race_model = DeepFace.build_model('Race')
print("finished Race Model")


# Calculates predictions for facial characteristics  


def facial_attr(frame, gender, age, emotion, race, dim, x, y):

    # specific settings for tailored for a specific model

    face_224 = functions.preprocess_face(img = frame, target_size = (224, 224), grayscale = False, enforce_detection = False, detector_backend = 'opencv')
    gray_img = functions.preprocess_face(img = frame, target_size = (48, 48), grayscale = True, enforce_detection = False, detector_backend = 'opencv')

    # Stores Results

    results =  {}

    # ------------EMOTION PREDICTION--------------

    if emotion:
        
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_predictions = emotion_model.predict(gray_img)[0,:]
        sum_of_predictions = emotion_predictions.sum()

        mood_items = []
        for i in range(0, len(emotion_labels)): 
            mood_item = []
            emotion_label = emotion_labels[i]
            emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
            mood_item.append(emotion_label)
            mood_item.append(emotion_prediction)
            mood_items.append(mood_item)

        emotion_df = pd.DataFrame(mood_items, columns = ["emotion", "score"])
        emotion_df = emotion_df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)
        top_emotion = emotion_df['emotion'][0]
        results['emotion'] = top_emotion


    #------------AGE PREDICTION-------------------                
    
    if age:
        
        print('it got here')
        age_predictions = age_model.predict(face_224)[0,:]
        apparent_age = Age.findApparentAge(age_predictions)
        results['age'] = apparent_age


    #---------GENDER PREDICTION----------------------

    if gender:

        gender_prediction = gender_model.predict(face_224)[0,:]

        if np.argmax(gender_prediction) == 0:
            face_gender = "W"
        elif np.argmax(gender_prediction) == 1:
            face_gender = "M"  

        results['gender'] = face_gender


    #-----------RACE PREDICTION (FIX THIS)----------------------

    if race == 1:
        
        race_predictions = race_model.predict(face_224)[0,:]
        race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
        sum_of_predictions = race_predictions.sum()
        dominate_race = race_labels[np.argmax(race_predictions)]
        results['race'] = dominate_race

    #-------------Outputs Results-------------------------------

    if results:

        y_index_counter = 20
        for key in results.keys():
            
            if key != "age":
                cv2.putText(frame, '{0}: {1}'.format(key.capitalize(),  results[key]), (x, y - y_index_counter), 0, 0.75, (255,255,255), 2)
            else:
                cv2.putText(frame, '{0}: {1:.2f}'.format(key.capitalize(),  results[key]), (x, y - y_index_counter), 0, 0.75, (255,255,255), 2)
            
            y_index_counter += 10

    
    #Return Frame

    return frame





# while video is running

def gen():

    #gloval attributes needed 

    global race, gender, age, emotion

    #constant variables needed for models

    size = 416
    iou = 0.45
    score = 0.50
    detector_backend = 'opencv'
    nms_max_overlap = 1.0
    frame_num = 0

    while True:

        # reads each frame so we can add detections

        success, frame = camera.read()

        if not success:
            
            print('Video has ended or failed, try a different video format!')
        
        else: 

            # change color to help improve accuracy

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            # re-adjust image size to keep it consistent

            print('Frame #: ', frame_num)
            image_data = cv2.resize(frame, (size, size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            # keep track of frames for FPS

            frame_num +=1

            # yolov4 batch information

            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]


            # Selects boxes with high IOU 

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50, 
                max_total_size=50,
                iou_threshold=iou,
                score_threshold=score
            )

            # convert data to numpy arrays and slice out unused elements

            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # classes we are using (There are many other classes) 
            
            allowed_classes = ['person']

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)

            # delete detections that are not in allowed_classes
            
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            # Call the tracker
            
            tracker.predict()
            tracker.update(detections)

            # update tracks

            for track in tracker.tracks:
                
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                
                bbox = track.to_tlbr()
                class_name = track.get_class()
                
            # draw bbox on screen

                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]

                
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            # Facial Features Prediction and Detection

                try:
                    faces_in_frame = FaceDetector.detect_faces(face_detector, detector_backend, frame, align = True)
                except:
                    faces_in_frame = []

                for face, dim in faces_in_frame:

                    x = dim[0]
                    y = dim[1]
                    w = dim[2]
                    h = dim[3]
                    
                    # Creates a rectangle around the face

                    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 1)

                    gender = gender
                    age = age
                    emotion = emotion
                    race = race

                    # Facial Prediction Function

                    frame = facial_attr(frame, gender, age, emotion, race, dim, int(bbox[0]), int(bbox[1]))


            # Calculate frames per second of running detections

            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # output video frame

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()  

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    


# HOMEPAGE

@app.route('/')
def index():
    return render_template('index.html')


# CARRIES OUT THE VIDEO PROCESS

@app.route('/video', methods=["GET", "POST"])
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


# COLLECTS THE USER INPUTS AND MAKE CHANGES

@app.route('/requests', methods=['GET', 'POST'])
def tasks():

    if request.method == "POST":
        
        if request.form.get('gender') == 'gender':
            global gender
            if gender == 0:
                gender=1
            else:
                gender=0
    

        elif  request.form.get('emotion') == 'emotion':
            global emotion
            if emotion == 0:
                emotion=1
            else:
                emotion=0
        

        elif  request.form.get('age') == 'age':
            global age
            if age == 0:
                age=1
            else:
                age=0

        elif  request.form.get('race') == 'race':
            global race
            if race == 0:
                race=1
            else:
                race=0 

    return render_template("index.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    