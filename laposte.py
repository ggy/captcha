'''
Created on 2 janv. 2014

@author: glagasse
'''

import re
import numpy as np
from matplotlib import pyplot as plt


import mechanize
import BeautifulSoup

import cv
import cv2
import pytesser
import oct2py
import Image

br = mechanize.Browser()
reponse = br.open("https://compte.laposte.net/inscription/etape1.do")
File = "."

Soup = BeautifulSoup
XML = Soup.BeautifulSoup(reponse.get_data())

img = XML.find('img', id='jcaptcha_img')
image_reponse = br.open_novisit(img['src'])
image = image_reponse.read()
fichier = file(File+'Poste.jpeg', 'w')
fichier.write(image)
fichier.close()




def OctaveGray():
    oc = oct2py.Oct2Py()
    oc.Img = oc.imread('Poste.jpeg')
    oc.Imgg = oc.rgb2gray(oc.Img)
    oc.imwrite(oc.Imgg, 'PosteGray.jpeg')

def CVGray():
    image = cv.LoadImage(File+"Poste.jpeg", cv.CV_LOAD_IMAGE_GRAYSCALE)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cv.SaveImage('PosteGray.jpeg', gray_image)

def TutoGray():
    import cv2
    image = cv2.imread(File+"Poste.jpeg")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(File+'gray_image.jpeg',gray_image)

def Histogramme(File):
    
    img = cv2.imread(File)
    h = np.zeros((300,256,3))
     
    bins = np.arange(256).reshape(256,1)
    color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([img],[ch],None,[256],[0,256])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.column_stack((bins,hist))
        cv2.polylines(h,[pts],False,col)
     
    h=np.flipud(h)
     
    cv2.imshow('colorhist',h)
    cv2.waitKey(0)
    
    return hist_item

def Contour(File):
        
    img = cv2.imread(File,0)
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
     
    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
                                                                        
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
     
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
     
    cv2.imshow("skel",skel)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def CouleurChoix(File, K, Boolean = False):
     
    img = cv2.imread(File)
    Z = img.reshape((-1,3))
     
    # convert to np.float32
    Z = np.float32(Z)
     
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    ret,label,center = cv2.kmeans(Z,K,bestLabels=None, criteria=criteria,attempts=10,flags=cv2.KMEANS_RANDOM_CENTERS)
     
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    gray_image = cv.fromarray(res2)
    cv.SaveImage('PosteGrayKmeans.jpeg', gray_image) 
    if Boolean == True:
        cv2.imshow('res2',res2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def AnalyseText():
    CouleurChoix(File+'gray_image.jpeg', 3, False)
    image = Image.open(File+'PosteGrayKmeans.jpeg')
    text = pytesser.image_to_string(image) 
    
    return text
    
TutoGray()

taille = 0
i = 0
text_valide = None
text = ""
chn_mdp = "[A-Za-z]{5,7}"
exp_mdp = re.compile(chn_mdp)

while (text_valide == None and i<= 3) :
    text = AnalyseText()
    text_valide = exp_mdp.search(text)

    print text_valide
    print text
    i = i + 1
    
#print Histogramme(File+'gray_image.jpeg')
#Contour(File+'gray_image.jpeg')

print "Resultat"
if  text_valide == None:
    print "Invalide"
else:
    print "valide"
print (text)