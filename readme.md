# License Plate Detection

Built specifically to work with a raspberry pi

training data taken from roboflow

## Activate virtual environment
`source yolov11_amd_env/bin/activate`

## Install dependencies
`pip install -r requirements.txt`

## (Optional if cache is small)
`pip install --no-cache-dir -r requirements.txt`

## Download dataset
`python dl.py`

## Train Model
`python train_model.py`


## Tech Stack

### Hardware:
- Raspberry Pi 5
- Raspberry Pi Camera v2.1

### Software:
- YOLOv11
- EasyOCR

### Hans Specifics
- Trained on a RX 6600XT, packages would be changed under an NVDA GPU
- fedora linux

verify using `python verify.py`


Normal test_model.py: 65.51 seconds
Multithread test_multithread.py: 21.95 seconds