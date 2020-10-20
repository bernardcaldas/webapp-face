import streamlit as st
import urllib.request
from PIL import Image
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import cv2
from base64 import b64decode
import io
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)


def draw_image_with_boxes(filename, result_list):

  # load the image
  	data = pyplot.imread(filename)
  # plot the image
  	pyplot.imshow(data)
  # get the context for drawing boxes
  	ax = pyplot.gca()
	
  	# plot each box

  	for result in result_list:

		  x, y, width, height = result['box']
		  rect = Rectangle((x, y), width, height, fill=False, color='red')
    	  # draw the box
		  ax.add_patch(rect)
  			# show the plot
  			#pyplot.show()





def main():


	st.title("Face Detection.")
	st.subheader("App to predict faces in images")
	image_file = st.file_uploader("Upload your image",type=['jpg','png','jpeg'])

	if image_file is not None:

		our_image = Image.open(image_file)
		st.text("Original Image")
    	#st.write('You selected `%s`' % our_image)
		st.image(our_image)
		img_array = np.array(our_image)
		cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))



	if st.button('predict'):

		# load image from file
		#pixels = pyplot.imread(our_image)
		#pixels = Image.open(filename)
		filename = 'out.jpg'
      	#src = cv2.imread(filename)
		src = cv2.imread('out.jpg')
		pixels = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
      	# create the detector, using default weights
		detector = MTCNN()
      	# detect faces in the image
		faces = detector.detect_faces(pixels)
      	# display faces on the original image
		draw_image_with_boxes(filename, faces)
	  	#st.success("Found {} faces".format(len(faces)))
		#st.text("Prediction")
		st.success("Found {} faces".format(len(faces)))
		st.pyplot()
      	#print('faces detected:', len(faces))






if __name__ == '__main__':
    main()