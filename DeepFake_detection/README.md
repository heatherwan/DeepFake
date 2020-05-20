## User Information after modification
# Datasets:
**Source:**
- Total input images from FaceForensics++ dataset are in `all_faces.zip` in google drive.

- Or use fake video we generated with our own model. 
    Before put into folder, we need to extract face from frame with `extract_face.py`. 
    
    Firstly, get frame use 
    
    `ffmpeg -i input_video.mp4 -vf h264 -vf fps=1/2 %05d.jpg` (extract frames from video  every 2 secs). 
    
    or `ffmpeg -i input_video.mp4 -vf scale=256:256 %05d.jpg` (extract 30 frames from video for every sec).
    
    Secondly, put these frames in `data_for_extract/input_images` and then run `extract_face.py`. The output will be stored in `data_for_extract/extract/faces`
        
**Organization:**
    
All images folders has 2 class(df/real): 
- `train_images` image to train the classification model
- `valid_images` image to test the classification accuracy
- `test_images` additional image to varify the result of our model


- `test_videos` you can put video and judge with all frames (aggrgate with average)

# Model Training and Testing

**1. Pre-trained Model**

All pre-trained model are stored in `weights`
- `Meso4_F2F` : trained model by paper
- `train_model.h5` : trained model with 300 frames from FaceForensics++ dataset
- `best_weights.hdf5` : trained model with 5000 frames from FaceForensics++ dataset.


**2. `train.py`**

1. choose pre-trained model 
2. change number of train and validation steps (based on image amounts, should be lower than total number of images input) 
3. training logs will store in `output/log.csv`; best model will store in `weights/best_model.hdf5`

**3. `test.py`**
1. test frames
incorrect classified images will log in `output/incorrect_test_images.txt`.
2. test videos: result shows in console.

# Acknowledgement

This code is heavily borrowed from [MesoNet](https://github.com/DariusAf/MesoNet)
For more detailed environment setup, please refer to the MesoNet repo. 
