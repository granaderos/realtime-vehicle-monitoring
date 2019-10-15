from django.shortcuts import render
from datetime import datetime
from .models import PlateNumberImage
import requests
from django.http import JsonResponse

import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image

# Create your views here.
def render_interface(request):
    return render(request, 'interface/index.html', {"message": ""})

def save_and_convert_image(request):
    print(request.FILES['fileinput'])
    image = request.FILES["fileinput"]
    name = str(datetime.now)

    plate_number_image = PlateNumberImage.objects.create(image=image, name=name)
    plate_number_image.save()

    # convert plate number image to text

    img = cv2.imread(str(plate_number_image.image), cv2.IMREAD_COLOR)
    
    img = cv2.resize(img, (620,480))

    #img = cv2.resize(img,None,fx=0.5,fy=0.5, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale



    gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise



    edged = cv2.Canny(gray, 5, 200) #Perform Edge detection
    # edged = cv2.Canny(gray, 1, 500) #Perform Edge detection

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse = True)[:10]
    screenCnt = None



    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    

    if screenCnt is None:
        detected = 0
        text = "NONE"
        print("No contour detected")
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
        # Masking the part other than the number plate
        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
        # new_image = cv2.drawContours(mask,[screenCnt],0, 255, 0,)
        new_image = cv2.bitwise_and(img,img,mask=mask)


        # Now crop
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx+1, topy:bottomy+1]

        
        config = ('-l eng --oem 1 --psm 3')
        # config='--psm 11'
        #Read the number plate
        text = pytesseract.image_to_string(Cropped, config=config)
    print("Detected Number is:", text)
    if text is not None and text != "NONE" and text != "":
        server_ip = "http://192.168.43.84:8000/api/entries/"

        data = dict(
            plate_number=text,
        )

        r = requests.post(server_ip, data=data)
        print(r)
        message = "Machine was able to read the plate number " + text + "."
        return render(request, 'interface/index.html', {"message": message})


    else:
        message = "Machine was not able to detect the plate number."

        # return JsonResponse({"plate_number": text})
        return render(request, 'interface/index.html', {"message": message})


    # cv2.imshow('Vehicle',img)
    # cv2.imshow('Plate Number',Cropped)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # server_ip = "192.168.43.84:8000/api/entries/"


