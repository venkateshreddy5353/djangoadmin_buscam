from datetime import datetime
import numpy as np
import dlib
from scipy.spatial import distance
from imutils import face_utils
import imagezmq
import argparse
import imutils
from PIL import Image
import cv2
import threading
import traceback
from time import sleep

class BusCam(object):   
    def __init__(self):
        self.prototxt = "MobileNetSSD_deploy.prototxt"
        self.model = "MobileNetSSD_deploy.caffemodel"
        self.confidence = 0.2

        self.thresh = 0.25
        self.frame_check = 20
        self.belt = False
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        self.flag=0

        # initialize the list of class labels MobileNet SSD was trained to
        # detect, then generate a set of bounding box colors for each class
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor","mobile","iPhone"]

        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)

        # initialize the consider set (class labels we care about and want
        # to count), the object count dictionary, and the frame  dictionary
        self.CONSIDER = set(["person","sofa","bottle"])
        #objCount = {obj: 0 for obj in self.CONSIDER}
        print("[INFO] detecting: {}...".format(", ".join(obj for obj in self.CONSIDER)))

        # initialize the dictionary which will contain information regarding
        # when a device was last active, then store the last time the check
        # was made was now
        self.frameDict = {}
        self.lastActive = {}
        self.lastActiveCheck = datetime.now()

        # stores the estimated number of Pis, active checking period, and
        # calculates the duration seconds to wait before making a check to
        # see if a device was active
        self.ESTIMATED_NUM_PIS = 2
        self.ACTIVE_CHECK_PERIOD = 10
        self.ACTIVE_CHECK_SECONDS = self.ESTIMATED_NUM_PIS * self.ACTIVE_CHECK_PERIOD

        self._active_thread = threading.Thread(target = self._active_check, args=())
        self._active_thread.daemon = True
        self._active_thread.start()

        self._object_frame = np.zeros([225,300,3], np.uint8)
        self._drowsy_frame = np.zeros([225,300,3], np.uint8)
        self._seatbelt_frame = np.zeros([225,300,3], np.uint8)
        self._frameDict = {'picroft':['picroft', np.zeros([225,300,3], np.uint8),np.zeros([225,300,3], np.uint8),np.zeros([225,300,3], np.uint8),np.zeros([225,300,3], np.uint8),np.zeros([225,300,3], np.uint8),np.zeros([225,300,3], np.uint8)],'pi-vaibhav':['pi-vaibhav', np.zeros([225,300,3], np.uint8),np.zeros([225,300,3], np.uint8),np.zeros([225,300,3], np.uint8),np.zeros([225,300,3], np.uint8),np.zeros([225,300,3], np.uint8),np.zeros([225,300,3], np.uint8)]}

        self._stop = False
        self._data_ready = threading.Event()
        self._thread = threading.Thread(target = self._run, args=())
        self._thread.daemon = True
        self._thread.start()

        self._object_ready = threading.Event()
        self._object_thread = threading.Thread(target = self._object_detection, args=())
        self._object_thread.daemon = True
        self._object_thread.start()

        self._drowsy_ready = threading.Event()
        self._drowsy_thread = threading.Thread(target = self._drowsy_detection, args=())
        self._drowsy_thread.daemon = True
        self._drowsy_thread.start()

        self._seatbelt_ready = threading.Event()
        self._seatbelt_thread = threading.Thread(target = self._seatbelt_detection, args=())
        self._seatbelt_thread.daemon = True
        self._seatbelt_thread.start()

    def receive(self,timeout = 60):
        flag = self._data_ready.wait(timeout = timeout)
        if not flag:
            #raise TimeoutError("Timeout while reading from subscriber")
            print("Timeout while reading from rpi")
        self._data_ready.clear()
        return self._data

    def _run(self): 
        receiver = imagezmq.ImageHub()
        while not self._stop:
            #(self.rpi_recv, self.frame_recv) = self.imageHub.recv_image()
            self._data = receiver.recv_image()
            receiver.send_reply(b'OK')
            self._data_ready.set()
        receiver.close()

    def close(self): 
        self._stop = True

    def detect_object(self, frame, rpiName):
        if self._object_ready.is_set():
            #self._object_frame = frame
            self._frameDict[rpiName][1] = frame
            self._object_ready.clear()
        #return self.object_frame
        return self._frameDict[rpiName][2]

    def _object_detection(self):
        while True:
            for rpiName in self._frameDict.keys():
                _frame = self._frameDict[rpiName][1]
                (h, w) = _frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(_frame, (300, 300)),0.007843, (300, 300), 127.5)
                # pass the blob through the network and obtain the detections and predections
                self.net.setInput(blob)
                detections = self.net.forward()
                # reset the object count for each object in the CONSIDER set
                objCount = {obj: 0 for obj in self.CONSIDER}
                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with the predictions
                    confidence = detections[0, 0, i, 2]
                    # filter out weak detections by ensuring the confidence is
                    # greater than the minimum confidence
                    if confidence > self.confidence:
                        # extract the index of the class label from the
                        # detections
                        idx = int(detections[0, 0, i, 1])
                        # check to see if the predicted class is in the set of
                        # classes that need to be considered
                        if self.CLASSES[idx] in self.CONSIDER:
                            # increment the count of the particular object
                            # detected in the frame
                            objCount[self.CLASSES[idx]] += 1
                            # compute the (x, y)-coordinates of the bounding box
                            # for the object
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            cv2.rectangle(_frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                    cv2.putText(_frame, rpiName, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    label = ", ".join("{}: {}".format(obj, count) for (obj, count) in objCount.items())
                    cv2.putText(_frame, label, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)
                self._frameDict[rpiName][2] = _frame
            self._object_ready.set()

    def detect_drowsy(self, frame, rpiName):
        if self._drowsy_ready.is_set() :
            #self._drowsy_frame = frame
            self._frameDict[rpiName][3] = frame
            self._drowsy_ready.clear()
        #return self.drowsy_frame
        return self._frameDict[rpiName][4]

    def eye_aspect_ratio(self, eye): 
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def _drowsy_detection(self):
        while True:
            for rpiName in self._frameDict.keys():
                _frame = self._frameDict[rpiName][3]
                gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
                subjects = self.detect(gray, 0)

                for subject in subjects:
                    shape = self.predict(gray, subject)
                    shape = face_utils.shape_to_np(shape)#converting to NumPy Array
                    leftEye = shape[self.lStart:self.lEnd]
                    rightEye = shape[self.rStart:self.rEnd]
                    leftEAR = self.eye_aspect_ratio(leftEye)
                    rightEAR = self.eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(_frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(_frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    if ear < self.thresh:
                        self.flag += 1
                        print(self.flag)
                        cv2.putText(_frame, str(self.flag), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        if self.flag >= self.frame_check:
                            cv2.putText(_frame, "Sleepy_Alert-Driver_is_Drowsy", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            print ("Drowsy")
                    else:
                        self.flag = 0
                self._frameDict[rpiName][4] = _frame
            self._drowsy_ready.set()

    def detect_seatbelt(self, frame, rpiName):
        if self._seatbelt_ready.is_set() :
            #self._seatbelt_frame = frame
            self._frameDict[rpiName][5] = frame
            self._seatbelt_ready.clear()
        #return self.seatbelt_frame
        return self._frameDict[rpiName][6]
    def slope(self,a,b,c,d):
        return (d - b)/(c - a)
    def _seatbelt_detection(self):
        while True:
            for rpiName in self._frameDict.keys():
                _frame = self._frameDict[rpiName][5]

                blur = cv2.blur(_frame, (1, 1))
                # Converting Image To Edges
                edges = cv2.Canny(blur, 50, 400)
                # Previous Line Slope
                ps = 0

                # Previous Line Co-ordinates
                px1, py1, px2, py2 = 0, 0, 0, 0
                # Extracting Lines
                lines = cv2.HoughLinesP(edges, 1, np.pi/270, 30, maxLineGap = 20, minLineLength = 170)
                # extract the confidence (i.e., probability) associated with
                # the prediction
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]# Co-ordinates Of Current Line
                        s = self.slope(x1,y1,x2,y2)# Slope Of Current Line
                        # If Current Line's Slope Is Greater Than 0.7 And Less Than 2
                        if ((abs(s) > 0.7) and (abs (s) < 2)):
                            # And Previous Line's Slope Is Within 0.7 To 2
                            if((abs(ps) > 0.7) and (abs(ps) < 2)):
                                # And Both The Lines Are Not Too Far From Each Other
                                if(((abs(x1 - px1) > 5) and (abs(x2 - px2) > 5)) or ((abs(y1 - py1) > 5) and (abs(y2 - py2) > 5))):
                                    # Plot The Lines On "beltframe"
                                    cv2.line(_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                    cv2.line(_frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
                                    print ("Belt Detected")
                                    self.belt = True
                                    # Otherwise Current Slope Becomes Previous Slope (ps) And Current Line Becomes Previous Line (px1, py1, px2, py2) 
                        ps = s 
                        px1, py1, px2, py2 = line[0]
                if self.belt == False:
                    #print("No Seatbelt detected")  
                    cv2.putText(_frame,"No Seatbelt detected", (00, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                self._frameDict[rpiName][6] = _frame
            self._seatbelt_ready.set()
    #---------------------------------------------------------------------------------------------------------------
    def get_frame(self):
        while True:
            framev = np.ones([2,1500,3], np.uint8)
            (rpiName, frame) = self.receive()
            if rpiName not in self.lastActive.keys(): 
                print("[INFO] receiving data from {}...".format(rpiName))
            self.lastActive[rpiName] = datetime.now()
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            self.frameDict[rpiName] = frame
            for (rpiName,frame) in self.frameDict.items():
                frame1 = frame[:,:300,:]
                frame2 = frame[:,300:600,:]
                frame3 = frame[:,600:900,:]
                frame4 = frame[:,900:1200,:]
                frame5 = frame[:,1200:1500,:]
            
                frame1 = self.detect_object(frame1,rpiName)
                
                frame2 = self.detect_drowsy(frame2, rpiName)

                frame3 = self.detect_seatbelt(frame3, rpiName)
                # put detection functions here
            
                frameh = np.concatenate([frame1,frame2,frame3,frame4,frame5],axis=1)
                framev = np.append(framev, frameh, axis=0)
            framenc = cv2.imencode('.jpg',framev)[1]
            return framenc.tobytes()
            #yield b'--framenc\r\nContent-Type:image/jpeg\r\n\r\n'+framenc.tobytes()+b'\r\n'
            
    def _active_check(self):
        while True:
            if (datetime.now() - self.lastActiveCheck).seconds > self.ACTIVE_CHECK_SECONDS:
                print("[INFO] performing active check")
                for (rpiName, ts) in list(self.lastActive.items()):
                    if (datetime.now() - ts).seconds > self.ACTIVE_CHECK_SECONDS:
                        print("[INFO] lost connection to {}".format(rpiName))
                        self.lastActive.pop(rpiName)
                        self.frameDict.pop(rpiName)
                self.lastActiveCheck = datetime.now()
def gen(camera):
    framesn = camera.get_frame()
    yield (b'--framesn\r\n'b'Content-Type: image/jpeg\r\n\r\n' + framesn + b'\r\n\r\n')

# do a bit of cleanup
cv2.destroyAllWindows()