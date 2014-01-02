'''
Created on 2 janv. 2014

@author: glagasse
'''

import mechanize
#from PIL import Image
from operator import itemgetter
import lxml
import BeautifulSoup
import sys
import imghdr
import cv
import cv2
from pygame import surfarray, image, display
import pygame
import numpy #important to import
#import pytesser

br = mechanize.Browser()
reponse = br.open("https://compte.laposte.net/inscription/etape1.do
")


Soup = BeautifulSoup
XML = Soup.BeautifulSoup(reponse.get_data())

img = XML.find('img', id='jcaptcha_img')
image_reponse = br.open_novisit(img['src'])
image = image_reponse.read()
fichier = file('Poste.jpeg', 'w')
fichier.write(image)
fichier.close
#print pytesser.image_to_string(imagePIL)     # Run tesseract.exe on image
#print pytesser.image_file_to_string('Poste.jpg')