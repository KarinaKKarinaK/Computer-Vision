import cv2
import mediapipe as mp
import time
#This script is to be used as a module

class handDetector():
    def __init__(self,mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode #Creating a variable of the object (object -> 'self') providing it with teh vaue of teh mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return img #To use an object in all methods of a class You have to first put self. and tehn the object

    def findPosition(self, img, handNo = 0, draw = True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo] 

            for id, lm in enumerate(myHand.landmark):
                    #print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h) #Position of teh center
                    #print(id, cx, cy)
                    lmList.append([id, cx, cy])
                    if draw:
                    #if id == 4: #The id is one of the points n teh hand, tehere are 21 points, 0 is the center at the bone, 4 is the thumb tip
                        cv2.circle(img, (cx, cy), 7, (0, 0, 0), cv2.FILLED)

        return lmList
                
def main():
    pTime = 0 #Previous Time
    cTime = 0 #Current Time
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4]) #now You can print teh position at any index, teh index being teh number assosciated with teh point

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime #The previous time becomes the current time

        #Putting the time on teh screen
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3, (255, 255, 255), 2) 
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__": #This means 'if we are running this script then do this()'
    main()