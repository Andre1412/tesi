import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import csv
import os

import pandas as pd
import numpy as np
model_path = './models/pose_landmarker_full.task'

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def make_detections(source):
  BaseOptions = mp.tasks.BaseOptions
  PoseLandmarker = mp.tasks.vision.PoseLandmarker
  PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  options = PoseLandmarkerOptions(
      base_options=BaseOptions(model_asset_path=model_path),
      running_mode=VisionRunningMode.IMAGE)

  with PoseLandmarker.create_from_options(options) as landmarker:
    print(source)
    mp_image = mp.Image.create_from_file(source)
    pose_landmarker_result = landmarker.detect(mp_image)
  annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
  cv2.imshow("image",cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

  return pose_landmarker_result


with  open('coords.csv', 'w', newline='')  as coords:
  writer = csv.writer(coords, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  

  firstTime=True
  for emotion in os.listdir("./dataset"):
    if not emotion.startswith('.'):
      for img in os.listdir("./dataset/"+ emotion):
          if not img.startswith('.'):
            results=make_detections("./dataset/"+ emotion + "/" + img)

            if firstTime:
                fields = ["class"]
                for x in range(len(results.pose_landmarks[0])):
                  fields +=["x{}".format(x), "y{}".format(x),"z{}".format(x),"v{}".format(x)]
                writer.writerow(fields)
                firstTime=False
            row = [emotion]
            for x in results.pose_landmarks[0]:
              row.extend([x.x, x.y, x.z, x.visibility])
            writer.writerow(row)

"""reader=csv.DictReader(coords, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  print("qui")
  for row in reader:
    print("\naaaa",row)
    if row:"""
cv2.waitKey(0)
cv2.destroyAllWindows()
  
