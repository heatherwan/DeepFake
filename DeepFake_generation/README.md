# DeepFake Generation Model with mask implementation

This project is based on the pytorch CycleGAN implementation from  
**CycleGAN: [Project](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) |  [Paper](https://arxiv.org/pdf/1703.10593.pdf) |**

**==About Implementation==**

According to the original model, the following scripts are created to adapt the mask implementation:

**1. CreateMask.py**

- Use **[face_alignment](http://github.com/1adrianb/face-alignment)** library to extract face part and create mask object from image. 
  
**2. prepare_training_data_and_train.py**

- Training model with mask setting. Face part weight([1,255])(eg. eyes, mouth,...) and Face ratio([1,..])

- The result model will stored in **[checkpoints](checkpoints/)** under the corresponding experiment name folder.

**3. run_and_generate_fake_video.py**

- Extract frames to test model(generate fake frame) and convert into videos. 

- There will be 2 videos generated in **[datasets](datasets/)** under the corresponding experiment name folder, one with RealA ending is one with correct frame sequences. 

- If the other direction wanted, exchange the video and add args **--direction BtoA**

- Videos can be only fake itself as single (default) or with original frame showing at the same time (with **--frame double**).

**==About Dataset==**

- For running the code above, videos are need to be stored in **[datasets](datasets/)** under **[original_video](datasets/original_video).**

- Video with around 2 minutes and static background is preferable to get a better result.

***Example runing script are all stored in [Scripts](scripts/).**

***Information about the original project are all stored in [Docs](docs/) and [README_original](README_original.md).**







