import cv2
import numpy as np
import pafy

camera = cv2.VideoCapture(1)
camera.set(3,1920)
camera.set(4,1080)

#url = "https://www.youtube.com/watch?v=HqErSOEjHe4" # Fish
#url = "https://www.youtube.com/watch?v=icDwkh0kF0k" # Waves
url = "https://www.youtube.com/watch?v=yVEKfLGwH0Q" # Gull
url = "https://www.youtube.com/watch?v=n9AG43bwkos"

source = pafy.new(url)
best = source.getbest(preftype="mp4")

video = cv2.VideoCapture()
video.open(best.url)

while True:

  retV, frame = video.read()
  retC, image = camera.read()
  if retC == True and retV == True:

    h, w, depth = image.shape

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blured = cv2.medianBlur(grey, 9)

    # grap the contours
    ret, thresh = cv2.threshold(blured,180, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    # Generate mask
    mask = np.zeros((h, w), np.uint8)
    for cnt in contours:

        if 50 < cv2.contourArea(cnt) < 150000:

          M = cv2.moments(cnt)
          cX = int(M["m10"] / M["m00"])
          cY = int(M["m01"] / M["m00"])

          # block any reflections in the lower third
          if cY < 2*h/3:
            cv2.drawContours(mask,[cnt],0,255,-1)

    # resize the video frame to match the camera
    #frame = cv2.resize(frame[250:600, 350:1050], (w, h), interpolation = cv2.INTER_AREA)
    frame = cv2.resize(frame, (w, h), interpolation = cv2.INTER_AREA)

    # merge forground and background
    background = cv2.bitwise_and(frame, frame, mask = mask)
    foreground = cv2.bitwise_and(image, image, mask = 255 - mask)
    result = foreground + background
    
    cv2.imshow('Camera stream', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
