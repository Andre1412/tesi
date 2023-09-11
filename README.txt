To train load images in the dataset folder and organize it in subfolders based on the
emotion/pose you want to classify, then run:
    python3 ./train_holistic.py if you want to use the holistic model (face, pose and hands models)
    python3 ./train.py if you want to use only the pose model


To predict run:
    python3 ./body_pose_classification.py
Flags:
    -i "imgPath" run on images
    -w run on webcam (default)