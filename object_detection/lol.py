import numpy as np
import os
from shutil import copyfile
import xml.etree.ElementTree as ET
import cv2



p='images/test'
'''
for files in sorted(os.listdir(p)):
	if(files.split('.')[1]!='jpg'):
		tree = ET.parse(p+'/'+files)
		root = tree.getroot()
		for rank in root.iter('width'):
			new=int(int(rank.text)*(1/2))
			rank.text=str(new)
		for rank in root.iter('height'):
			new=int(int(rank.text)*(1/2))
			rank.text=str(new)

		for rank in root.iter('xmin'):
			new=int(int(rank.text)*(1/2))
			rank.text=str(new)
		for rank in root.iter('xmax'):
			new=int(int(rank.text)*(1/2))
			rank.text=str(new)
		for rank in root.iter('ymin'):
			new=int(int(rank.text)*(1/2))
			rank.text=str(new)
		for rank in root.iter('ymax'):
			new=int(int(rank.text)*(1/2))
			rank.text=str(new)


		tree.write(p+'/'+files)
'''

for files in sorted(os.listdir(p)):
	if(files.split('.')[1]=='jpg'):
		img=cv2.imread(p+'/'+files)
		img =cv2.resize(img, (0,0), fx=0.5, fy=0.5)  
		cv2.imwrite(p+'/'+files,img)







