Built specifically to work with a raspberry pi

training data taken from roboflow

1st: Activate virtual environment: `source yolov11_env/bin/activate` 

2nd:`pip install -r requirements.txt` or `pip install --no-cache-dir -r requirements.txt` if you're like me and accidentally installed your OS on a HDD instead of an SSD 

3rd: `python dl.py` - doing this will install the data from roboflow, create an API key before you do this and place it in a .env file


TechStack: 

Hardware: 
Raspberry pi 5
Raspberry pi camera v2.1

Software: 

Yolo v11
Easy OCR
