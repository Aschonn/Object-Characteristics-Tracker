import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import time
import re

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deepface import DeepFace
from deepface.extendedmodels import Age, Race
from deepface.commons import functions, realtime, distance as dst
from deepface.detectors import FaceDetector
    
def tracker(frame, actions = {}):
    
    
    face_detector = FaceDetector.build_model(detector_backend)

    detector_backend = 'opencv'
    num_of_faces = 0
    frame_count = 0
    time_per_face = []

    try:
        faces_in_frame = FaceDetector.detect_faces(face_detector, detector_backend, frame, align = True)
    except:
        faces_in_frame = []

    for face, dim in faces_in_frame:

        num_of_faces += 1

        x = dim[0]
        y = dim[1]
        w = dim[2]
        h = dim[3]
        
        # Creates a rectangle around the face

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,233,0), 1)

        # specific settings for tailored for a specific model

        face_224 = functions.preprocess_face(img = frame, target_size = (224, 224), grayscale = False, enforce_detection = False, detector_backend = 'opencv')
        gray_img = functions.preprocess_face(img = frame, target_size = (48, 48), grayscale = True, enforce_detection = False, detector_backend = 'opencv')


        # ------------EMOTION PREDICTION--------------

        if 'emotion_model' in actions:

            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            emotion_predictions = actions['emotion_model'].predict(gray_img)[0,:]
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
        
        #------------AGE PREDICTION-------------------

        if 'age_model' in actions:

            age_predictions = actions['age_model'].predict(face_224)[0,:]
            apparent_age = Age.findApparentAge(age_predictions)

        #---------GENDER PREDICTION----------------------

        if 'gender_model' in actions:

            gender_prediction = actions['gender_model'].predict(face_224)[0,:]

            if np.argmax(gender_prediction) == 0:
                gender = "W"
            elif np.argmax(gender_prediction) == 1:
                gender = "M"


        #-----------RACE PREDICTION (FIX THIS)----------------------

        if 'race_model' in actions:

            race_predictions = actions['race_model'].predict(face_224)[0,:]
            race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
            sum_of_predictions = race_predictions.sum()
            dominate_race = race_labels[np.argmax(race_predictions)]




    return {
        'emotion' : emotion_df['emotion'][0],
        'age'     : apparent_age,
        'race'    : dominate_race,
        'gender'  : gender
    }


        #-----------OUTPUT RESULTS------------------------

    #     font_color = (36,255,12)

    #     start_point = (x, y)
    #     end_point = (x + w, y - 70)

    #     background = cv2.rectangle(frame, start_point, end_point, (0, 0, 0), -1)
    #     cv2.imshow("img", background)

    #     cv2.putText(frame, 'Sex: {}'.format(gender), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1)
    #     cv2.putText(frame, 'Age: {:.2f}'.format(apparent_age), (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1)
    #     cv2.putText(frame, 'Emotion: {}'.format(emotion_df['emotion'][0]), (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1)
    #     cv2.putText(frame, 'Race: {}'.format(dominate_race), (x, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1)


    #     cv2.namedWindow('frame', cv2.WINDOW_FREERATIO)
    #     cv2.imshow('frame', frame)
    #     toc = time.time()

    #     if len(time_per_face) != 0:
    #         print("FPS OVERALL: {0:.2f} , TPF (Time per face): {1:.2f} # Faces: {2}".format(frame_count/(toc-tic), sum(time_per_face)/len(time_per_face), num_of_faces))
    #     else:		
    #         print("FPS OVERALL: {0:.2f}, # Faces: {1}".format(frame_count/(toc-tic), num_of_faces))

    #     # QUIT PROGRAM
        
    #     if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
    #         break

    # #kill open cv things
    # cap.release()
    # cv2.destroyAllWindows()




# def tracker(cap, face_detector, actions = {}):

#     # ADD TIME ASPECT TO APPLICATION? 

#     detector_backend = 'opencv'
#     enable_face_analysis = True
#     num_of_faces = 0
#     frame_count = 0
#     time_per_face = []

#     while(True):
        
#         ret, frame = cap.read()

#         frame_count += 1

#         if frame is None:
#             break

#         try:
#             faces_in_frame = FaceDetector.detect_faces(face_detector, detector_backend, frame, align = True)
#         except:
#             faces_in_frame = []

#         for face, dim in faces_in_frame:

#             num_of_faces += 1

#             x = dim[0]
#             y = dim[1]
#             w = dim[2]
#             h = dim[3]
            
#             # Creates a rectangle around the face

#             cv2.rectangle(frame, (x,y), (x+w,y+h), (255,233,0), 1)

#             # specific settings for tailored for a specific model

#             face_224 = functions.preprocess_face(img = frame, target_size = (224, 224), grayscale = False, enforce_detection = False, detector_backend = 'opencv')
#             gray_img = functions.preprocess_face(img = frame, target_size = (48, 48), grayscale = True, enforce_detection = False, detector_backend = 'opencv')

    
#             # ------------EMOTION PREDICTION--------------

#             if 'emotion_model' in actions:

#                 emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#                 emotion_predictions = actions['emotion_model'].predict(gray_img)[0,:]
#                 sum_of_predictions = emotion_predictions.sum()

#                 mood_items = []
#                 for i in range(0, len(emotion_labels)): 
#                     mood_item = []
#                     emotion_label = emotion_labels[i]
#                     emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
#                     mood_item.append(emotion_label)
#                     mood_item.append(emotion_prediction)
#                     mood_items.append(mood_item)

#                 emotion_df = pd.DataFrame(mood_items, columns = ["emotion", "score"])
#                 emotion_df = emotion_df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)
            
#             #------------AGE PREDICTION-------------------

#             if 'age_model' in actions:

#                 age_predictions = actions['age_model'].predict(face_224)[0,:]
#                 apparent_age = Age.findApparentAge(age_predictions)

#             #---------GENDER PREDICTION----------------------

#             if 'gender_model' in actions:

#                 gender_prediction = actions['gender_model'].predict(face_224)[0,:]

#                 if np.argmax(gender_prediction) == 0:
#                     gender = "W"
#                 elif np.argmax(gender_prediction) == 1:
#                     gender = "M"


#             #-----------RACE PREDICTION (FIX THIS)----------------------

#             if 'race_model' in actions:

#                 race_predictions = actions['race_model'].predict(face_224)[0,:]
#                 race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
#                 sum_of_predictions = race_predictions.sum()
#                 dominate_race = race_labels[np.argmax(race_predictions)]


#             #-----------OUTPUT RESULTS------------------------

#             font_color = (36,255,12)

#             start_point = (x, y)
#             end_point = (x + w, y - 70)

#             background = cv2.rectangle(frame, start_point, end_point, (0, 0, 0), -1)
#             cv2.imshow("img", background)

#             cv2.putText(frame, 'Sex: {}'.format(gender), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1)
#             cv2.putText(frame, 'Age: {:.2f}'.format(apparent_age), (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1)
#             cv2.putText(frame, 'Emotion: {}'.format(emotion_df['emotion'][0]), (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1)
#             cv2.putText(frame, 'Race: {}'.format(dominate_race), (x, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1)


#         cv2.namedWindow('frame', cv2.WINDOW_FREERATIO)
#         cv2.imshow('frame', frame)
#         toc = time.time()

#         if len(time_per_face) != 0:
#             print("FPS OVERALL: {0:.2f} , TPF (Time per face): {1:.2f} # Faces: {2}".format(frame_count/(toc-tic), sum(time_per_face)/len(time_per_face), num_of_faces))
#         else:		
#             print("FPS OVERALL: {0:.2f}, # Faces: {1}".format(frame_count/(toc-tic), num_of_faces))

#         # QUIT PROGRAM
        
#         if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
#             break

#     #kill open cv things
#     cap.release()
#     cv2.destroyAllWindows()