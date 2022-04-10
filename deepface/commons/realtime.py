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
from deepface.commons import recognition, tracker, distance as dst
from deepface.detectors import FaceDetector
from . import tracker

def controller(actions = [], file = None):

	#-------------LOADS THE MODELS -----------

	model_being_used = {}

	if 'emotion' in actions:
		emotion_model = DeepFace.build_model('Emotion')
		model_being_used["emotion_model"] = emotion_model
		print("Emotion model loaded")
	
	if 'age' in actions:

		age_model = DeepFace.build_model('Age')
		model_being_used["age_model"] = age_model
		print("Age model loaded")

	if 'gender' in actions:

		gender_model = DeepFace.build_model('Gender')
		model_being_used["gender_model"] = gender_model
		print("Gender model loaded")

	if 'race' in actions:
		race_model = DeepFace.build_model("Race")
		model_being_used["race_model"] = race_model
		print("Race model loaded")


	if file != None:
		recognition(file, model_being_used)
	else:
		tracker(model_being_used)

