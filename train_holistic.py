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
""" import joblib
 """
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

def make_detections(source):

  with mp_holistic.Holistic(
      static_image_mode=True,
      model_complexity=2,
      enable_segmentation=True,
      refine_face_landmarks=True) as holistic:
    print(source)
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
  df = pd.DataFrame(columns= fields);          
  for emotion in os.listdir("./dataset"):
    if not emotion.startswith('.'):
      for img in os.listdir("./dataset/"+ emotion):
        if not img.startswith('.'):
          results=make_detections("./dataset/"+ emotion + "/" + img)
          """ row = [emotion]
          size = 0
          if results.pose_landmarks:
            for x in results.pose_landmarks.landmark:
              row.extend([x.x, x.y, x.z, x.visibility])
              size +=4
          else :
            for x in range(33):
              row.extend([0,0,0,0]) 
          if results.face_landmarks:
            for x in results.face_landmarks.landmark:
              row.extend([x.x, x.y, x.z, x.visibility])
              size +=4
          else :
            for x in range(468):
              row.extend([0,0,0,0])  """            
          coordinate = [emotion]
          if results.pose_landmarks:
            for i,landmark in enumerate(results.pose_landmarks.landmark):
              if landmark is not None:
                  coordinate.append(landmark.x)
                  coordinate.append(landmark.y)
                  coordinate.append(landmark.z)
                  coordinate.append(landmark.visibility)
              else:
                  coordinate.append(None)
                  coordinate.append(None)
                  coordinate.append(None)
                  coordinate.append(None)
          else:
            for x in range(33):
              coordinate.append(None)
              coordinate.append(None)
              coordinate.append(None)
              coordinate.append(None)
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
                coordinate.append(None)
          else:
            for x in range(478):
              coordinate.append(None)
              coordinate.append(None)
              coordinate.append(None)
              coordinate.append(None)
          print(len(coordinate))
          df.loc[len(df.index)] = coordinate            
  df.to_csv('coords_test.csv', index=False)    

# Carica il modello pre-addestrato
#model = joblib.load('modello_addestrato.pkl')
write_csv()
#coords_data = pd.read_csv('coords.csv',sep=',')
#print(coords_data.head())

"""reader=csv.DictReader(coords, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  print("qui")
  for row in reader:
    print("\naaaa",row)
    if row:"""
cv2.waitKey(0)
cv2.destroyAllWindows()
  




""" 
# Crea un DataFrame vuoto
df = pd.DataFrame()

# Estrai e aggiungi le coordinate delle landmark al DataFrame
if results.pose_landmarks:
    coordinate = []
    for landmark in results.pose_landmarks.landmark:
        if landmark is not None:
            x = landmark.x
            y = landmark.y
        else:
            x = None
            y = None
        coordinate.append((x, y))
    df['PoseLandmark'] = coordinate

# Estrai e aggiungi le coordinate delle landmark del volto al DataFrame (se necessario)

    df['FaceLandmark'] = coordinate

# Salva il DataFrame in un file CSV
df.to_csv('landmarks.csv', index=False) """