import sys
from turtle import width

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


from sklearn.impute import SimpleImputer


import cv2
import numpy as np

import os

import pandas as pd

import joblib

mode = str(sys.argv[1])
mp_holistic = mp.solutions.holistic


def draw_landmarks_on_image(rgb_image, detection_result):
  annotated_image = np.copy(rgb_image)
  mp_drawing = mp.solutions.drawing_utils

  # Draw the face landmarks.
  mp_drawing.draw_landmarks(
    annotated_image, detection_result.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
  # Draw pose connections
  mp_drawing.draw_landmarks(annotated_image, detection_result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0,0,204), thickness=3, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(91,229,255), thickness=3, circle_radius=2)
                            ) 
  return annotated_image


def make_detections(source):
  #model = joblib.load('modello_addestrato_values.pkl')
  if mode=='-w':
    image=source
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  else:
    image = cv2.imread(source)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=True) as holistic:
    results = holistic.process(image)
  annotated_image = draw_landmarks_on_image(image, results)
  coordinate = []
  """     pose = list([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]) if results.pose_landmarks else list(np.array([[-1,-1,-1,0] for i in range(33)]))
    face = list([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.face_landmarks.landmark]) if results.face_landmarks else list(np.array([[-1,-1,-1,0] for i in range(478)]))
    print(pose)
    X=pd.DataFrame([pose + face])  
  """
  if results.pose_landmarks:
    for landmark in results.pose_landmarks.landmark:
      if landmark is not None:
          coordinate.append(landmark.x)
          coordinate.append(landmark.y)
          coordinate.append(landmark.z)
          coordinate.append(landmark.visibility)
      else:
          coordinate.append(None)
          coordinate.append(None)
          coordinate.append(None)
          coordinate.append(0)
  else:
    for x in range(33):
      coordinate.append(None)
      coordinate.append(None)
      coordinate.append(None)
      coordinate.append(0)
  if results.face_landmarks:
    for landmark in results.face_landmarks.landmark:
      if landmark is not None:
        coordinate.append(landmark.x)
        coordinate.append(landmark.y)
        coordinate.append(landmark.z)
        coordinate.append(landmark.visibility)
      else:
        coordinate.append(None)
        coordinate.append(None)
        coordinate.append(None)
        coordinate.append(0)
  else:
    for x in range(478):
      coordinate.append(None)
      coordinate.append(None)
      coordinate.append(None)
      coordinate.append(0)
  df = pd.read_csv('coords_impute.csv')
  
  fields = []
  for x in range(33):
    fields +=["px{}".format(x), "py{}".format(x),"pz{}".format(x),"pv{}".format(x)]
  for x in range(478):
    fields +=["fx{}".format(x), "fy{}".format(x),"fz{}".format(x),"fv{}".format(x)] 

  X=pd.DataFrame(columns=fields, data=[coordinate])
  imputer = SimpleImputer(missing_values = np.nan, strategy ='mean')
  imputer = imputer.fit(df.drop('class',axis=1))
  X = imputer.transform(X)

  body_language_class = "happy"
  body_language_prob = 0
  #body_language_class = model.predict(X)[0]
  #body_language_prob = model.predict_proba(X)[0]
  #body_language_class = ''

  """   size, _ = cv2.getTextSize(body_language_class + ": " +str(round(body_language_prob[np.argmax(body_language_prob)],2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, 10)
    width, height = size
    cv2.rectangle(annotated_image, (10, 10), (int(width), 150), (151, 207, 255), -1)
    cv2.putText(annotated_image, body_language_class+ ": " +str(round(body_language_prob[np.argmax(body_language_prob)],2)), (10, 100),cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (0,0,204), 10, cv2.LINE_AA)
  """  
  return results, annotated_image

def check_landmarks(emotion):
  for img in os.listdir("./dataset/"+ emotion):
    if not img.startswith('.'):
      results,img=make_detections("./dataset/"+ emotion + "/" + img)
      if not results.pose_landmarks and not results.face_landmarks:
        print(img)
      else:
        print('ok')

path=""
if mode == "-w": #webcam
  cap = cv2.VideoCapture(0)
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results, image=make_detections(image)
    cv2.imshow('MediaPipe Holistic', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
  cap.release()
  cv2.destroyAllWindows()
elif mode=='check':
  check_landmarks(str(sys.argv[2]))
else: #img
  if len(sys.argv) == 3 :
    path = str(sys.argv[2])
    print("------------",path)

  else :
    while not os.path.isfile(path):
      path = input("Path is invalid. Please enter valid img path: ")
  results, img=make_detections(path) 
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
  cv2.imshow("image",img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

