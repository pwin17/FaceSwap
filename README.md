# FaceSwap

Instructions to run PRNet:

1. Download the model weight from https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view
2. Place the file in ./PRNet/Data/net-data
3. Create a conda environment or venv with python2.7, tf 1.13 (gpu version), cv2 4.2.1, numpy and dlib.
4. In prnetWrapper.py, specify path to the source image/video and path to the destination image/video.
5. Run in command line: python prnetWrapper.py.