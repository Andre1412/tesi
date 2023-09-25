import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import csv
import os
import pprint

import pandas as pd
import numpy as np
import joblib

import seaborn as sns

import sys
from os.path import exists

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
mp_holistic = mp.solutions.holistic


def draw_landmarks_on_image(rgb_image, detection_result):
  annotated_image = np.copy(rgb_image)
  mp_drawing = mp.solutions.drawing_utils

  # Draw the pose landmarks.
  mp_drawing.draw_landmarks(
    annotated_image, detection_result.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
  # Draw pose connections
  mp_drawing.draw_landmarks(annotated_image, detection_result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                            ) 
  return annotated_image

  """     static_image_mode=True,
      model_complexity=2,
      enable_segmentation=True, """
def make_detections(source):
  with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
      refine_face_landmarks=True) as holistic:
    image = cv2.imread(source)
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  annotated_image = draw_landmarks_on_image(image, results)
  cv2.imshow("image",annotated_image)

  return results

def write_csv():
  fields = ["class"]
  for x in range(33):
    fields +=["px{}".format(x), "py{}".format(x),"pz{}".format(x),"pv{}".format(x)]
  for x in range(478):
    fields +=["fx{}".format(x), "fy{}".format(x),"fz{}".format(x),"fv{}".format(x)] 
  df = pd.DataFrame(columns= fields)         
  for emotion in os.listdir("./dataset"):
    if not emotion.startswith('.') and not emotion == 'test' and not emotion == 'check':
      print(emotion)
      for img in os.listdir("./dataset/"+ emotion) :
        if not img.startswith('.'):
          results=make_detections("./dataset/"+ emotion + "/" + img)
          if not results.pose_landmarks and not results.face_landmarks:
            print(img)
          coordinate = [emotion]
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
          df.loc[len(df.index)] = coordinate       
      selectedrows = df[df["class"]==emotion].drop("class", axis=1)
      """       for col in selectedrows.drop("class", axis=1):
              df.loc[df["class"]==emotion, col] = selectedrows[col].fillna(selectedrows[col].mean())
      """  
      imputer = SimpleImputer(missing_values = np.nan, strategy ='mean')
      imputer = imputer.fit(selectedrows)
      
      """       imputer = KNNImputer(n_neighbors=2)
            imputer.fit_transform(selectedrows)
      """      
      df.loc[df["class"]==emotion, selectedrows.columns] = imputer.transform(selectedrows)
  df.to_csv('coords_impute.csv', index=False)

def write_webcam_csv():
  cap = cv2.VideoCapture(0)
  fields = ["class"]
  for x in range(33):
    fields +=["px{}".format(x), "py{}".format(x),"pz{}".format(x),"pv{}".format(x)]
  for x in range(478):
    fields +=["fx{}".format(x), "fy{}".format(x),"fz{}".format(x),"fv{}".format(x)]
  # Initiate holistic model
  df = pd.DataFrame(columns=fields)
  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=True) as holistic:
    df = pd.DataFrame(columns=fields)
    while cap.isOpened():
      emotion='angry'
      coordinate = [emotion]
      ret, frame = cap.read()
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False      
      results = holistic.process(image)  
      image = draw_landmarks_on_image(image, results)
      cv2.imshow('Raw Webcam Feed', image)
      try:
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
        df.loc[len(df.index)] = coordinate    
      except:
        pass   
      if cv2.waitKey(10) & 0xFF == ord('q'):
          break
    selectedrows = df[df["class"]==emotion].drop("class", axis=1)
    imputer = SimpleImputer(missing_values = np.nan, strategy ='mean')
    imputer = imputer.fit(selectedrows)
    df.loc[df["class"]==emotion, selectedrows.columns] = imputer.transform(selectedrows)
    if not exists("./coords_webcam.csv"):
      df.to_csv('coords_webcam.csv', index=False)
    df.drop(df.index[0]).to_csv('coords_webcam.csv', mode='a', index=False, header=None)
  cap.release()
  cv2.destroyAllWindows()

#write_csv()
#write_webcam_csv()
# Carica il modello pre-addestrato se esiste
coords_data = pd.read_csv('coords_impute.csv',sep=',')
X = coords_data.drop('class', axis = 1).values
y = coords_data['class']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Addestramento di un classificatore SVM
classifier = MultiOutputClassifier(GradientBoostingClassifier(learning_rate=0.2))

classifier.fit(X_train, y_train)

# Effettua le predizioni sul set di test
y_pred = classifier.predict(X_test)

# Decodifica delle etichette predette
y_pred_decoded = label_encoder.inverse_transform(y_pred)
y_test_decoded = label_encoder.inverse_transform(y_test)

accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
report=classification_report(y_pred_decoded, y_test_decoded)
joblib.dump(classifier, 'modello_addestrato_multivalues.pkl')
print(report)
class_mapping = dict(zip(label_encoder.transform(y), y))
print(class_mapping)
