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
                            mp_drawing.DrawingSpec(color=(204,0,0), thickness=5, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(255,229,91), thickness=5, circle_radius=2)
                            ) 
  return annotated_image


def make_detections(source):
  model = joblib.load('modello_addestrato.pkl')
  if mode=='-w':
    image=source
  else:
    image = cv2.imread(source)
    print(source)
  with mp_holistic.Holistic(
      static_image_mode= mode=='-w',
      model_complexity=2,
      enable_segmentation=True,
      refine_face_landmarks=True) as holistic:
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  #annotated_image = draw_landmarks_on_image(image, results)
  annotated_image = image
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
          coordinate.append(-1)
          coordinate.append(-1)
          coordinate.append(-1)
          coordinate.append(0)
  else:
    for x in range(33):
      coordinate.append(-1)
      coordinate.append(-1)
      coordinate.append(-1)
      coordinate.append(0)
  if results.face_landmarks:
    for landmark in results.face_landmarks.landmark:
      if landmark is not None:
        coordinate.append(landmark.x)
        coordinate.append(landmark.y)
        coordinate.append(landmark.z)
        coordinate.append(landmark.visibility)
      else:
        coordinate.append(-1)
        coordinate.append(-1)
        coordinate.append(-1)
        coordinate.append(0)
  else:
    for x in range(478):
      coordinate.append(-1)
      coordinate.append(-1)
      coordinate.append(-1)
      coordinate.append(0)
  
  X=pd.DataFrame([coordinate])  
  """   imputer = SimpleImputer(missing_values = np.nan, strategy ='mean')
    imputer = imputer.fit(X)
    X = imputer.transform(X)
    print(X.head())
  """
  body_language_class = model.predict(X)[0]
  print(body_language_class)

  size, _ = cv2.getTextSize(body_language_class, cv2.FONT_HERSHEY_TRIPLEX, 8, 10)
  width, height = size
  cv2.rectangle(annotated_image, (10, 10), (int(width), 300), (255, 229, 204), -1)
  cv2.putText(annotated_image, body_language_class, (10, 300),cv2.FONT_HERSHEY_TRIPLEX, 8, (255, 0, 0), 10, cv2.LINE_AA)

  #cv2.imshow("image",annotated_image)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()
  return results, cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

def check_landmarks(emotion):
  for img in os.listdir("./dataset/"+ emotion):
    if not img.startswith('.'):
      results,img=make_detections("./dataset/"+ emotion + "/" + img)
      if not results.pose_landmarks and not results.face_landmarks:
        print("----------",img)

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
    results, ann_img=make_detections(image)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic', ann_img)
    if cv2.waitKey(1) & 0xFF == 27:
      break
  cap.release()
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
