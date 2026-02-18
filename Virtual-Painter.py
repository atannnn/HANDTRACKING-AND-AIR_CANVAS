import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

####################
# --- 1. SETTINGS ---
brushThickness = 15
markerThickness = 35
eraserThickness = 70

# Define Colors (BGR)
colorBlue = (230, 196, 97)
colorGray = (50, 50, 50)
colorRed = (73, 51, 219)
colorPurple = (235, 23, 94)

# --- 2. STATE MANAGEMENT ---
colorIndex = 0  
activeColor = colorBlue 
toolType = "brush" 

# *** CRITICAL FIX: Define drawColor here so it doesn't crash on start ***
drawColor = activeColor 
####################

# --- 3. LOAD IMAGES ---
folderPath = "Header"
myList = os.listdir(folderPath)
myList.sort() 
overlayList = []

print("Loading Header Images...")
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    
    if image is None:
        print(f"ERROR: Could not load image: {folderPath}/{imPath}")
        continue
        
    # *** CRITICAL FIX: Force resize to 1280x125 to prevent ValueError ***
    # This prevents the "broadcast input array" crash
    image = cv2.resize(image, (1280, 125)) 
    overlayList.append(image)

if len(overlayList) == 0:
    print("ERROR: No images found. Exiting.")
    exit()

print(f"Loaded {len(overlayList)} images.")
header = overlayList[0] 

# --- 4. SETUP CAM & DETECTOR ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# --- HELPER FUNCTION ---
def updateHeader(tType, cIndex):
    try:
        if tType == "eraser":
            return overlayList[8]
        elif tType == "brush":
            return overlayList[cIndex]      # 0 to 3
        elif tType == "marker":
            return overlayList[cIndex + 4]  # 4 to 7
    except IndexError:
        return overlayList[0]
    return overlayList[0]

while True:
    # 1. Import image
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    img = cv2.flip(img, 1)

    # Resize header if camera doesn't support 1280 width
    h, w, c = img.shape
    if header.shape[1] != w:
        header = cv2.resize(header, (w, 125))
    if imgCanvas.shape[1] != w:
        imgCanvas = cv2.resize(imgCanvas, (w, h))

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        # ---------------------------------------------------------
        # 3. SELECTION MODE
        # ---------------------------------------------------------
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0 
            
            # Check for click in header
            if y1 < 125:
                # --- A. COLOR SELECTION ---
                if 0 < x1 < 200:        # Blue
                    colorIndex = 0
                    activeColor = colorBlue
                elif 200 < x1 < 320:    # Gray
                    colorIndex = 1
                    activeColor = colorGray
                elif 320 < x1 < 440:    # Pink
                    colorIndex = 2
                    activeColor = colorRed
                elif 440 < x1 < 560:    # Purple
                    colorIndex = 3
                    activeColor = colorPurple

                if 0 < x1 < 560 and toolType == "eraser":
                    toolType = "brush" 

                # --- B. TOOL SELECTION ---
                elif 650 < x1 < 800:    # Brush
                    toolType = "brush"
                elif 800 < x1 < 950:    # Marker
                    toolType = "marker"
                elif 1050 < x1 < 1280:  # Eraser
                    toolType = "eraser"

                # --- C. UPDATE STATE ---
                header = updateHeader(toolType, colorIndex)
                
                if header.shape[1] != w:
                    header = cv2.resize(header, (w, 125))
                
                if toolType == "eraser":
                    drawColor = (0, 0, 0)
                else:
                    drawColor = activeColor

            # This line caused the crash before because drawColor wasn't defined
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # ---------------------------------------------------------
        # 4. DRAWING MODE
        # ---------------------------------------------------------
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if toolType == "brush":
                thick = brushThickness
            elif toolType == "marker":
                thick = markerThickness
            elif toolType == "eraser":
                thick = eraserThickness
            else:
                thick = brushThickness

            cv2.line(img, (xp, yp), (x1, y1), drawColor, thick)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thick)

            xp, yp = x1, y1

    # ---------------------------------------------------------
    # 5. MERGE & DISPLAY
    # ---------------------------------------------------------
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    
    if img.shape != imgInv.shape:
        imgInv = cv2.resize(imgInv, (img.shape[1], img.shape[0]))
        
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    headerHeight, headerWidth, _ = header.shape
    img[0:headerHeight, 0:headerWidth] = header
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
