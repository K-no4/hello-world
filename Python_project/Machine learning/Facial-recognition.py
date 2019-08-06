# import of needed libraries
import io
import cv2
import time
import math
import argparse
from PIL import Image, ImageDraw
from google.cloud import vision
from google.cloud.vision import types

# setting a variable to enable video capture
cap = cv2.VideoCapture(0)
# Calls to account to verify if the key is pressent
client = vision.ImageAnnotatorClient.from_service_account_json("/home/felixrunner/test-1.json")
# Change resolution to 1080p
def make_1080p():
	cap.set(3,1920)
	cap.set(4,1080)
# Change resolution ot 720p
def make_720():
	cap.set(3,1280)
	cap.set(4,720)

def detect_face(face_file, max_results=4):
	
    # call to read the image
    content = face_file.read()
    # process the image to be able to be 
    image = types.Image(content=content)

    return client.face_detection(image=image, max_results=max_results).face_annotations

#draws a box around detected faces
def highlight_faces(image, faces, output_filename):

    im = Image.open(image)
    draw = ImageDraw.Draw(im)
    for face in faces:
        box = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')
        # drawing box with confidence to detecting a face 
        draw.text(((face.bounding_poly.vertices)[0].x,
                   (face.bounding_poly.vertices)[0].y - 30),
                  str(format(face.detection_confidence, '.3f')) + '%',
                  fill='#FF0000')
    # Saves the changes done by the funtion to the file
    im.save(output_filename)


def main(input_filename, output_filename, max_results):
	with open(input_filename, 'rb') as image:
		faces = detect_face(image, max_results)
		print('Found {} face{}'.format(
			len(faces), '' if len(faces) == 1 else 's'))

		print('Writing to file {}'.format(output_filename))
        # Reset the file pointer, so we can read the file again
		image.seek(0)
		highlight_faces(image, faces, output_filename)

def localize_objects(path):
	with open(path, 'rb') as image_file:
		content = image_file.read()
		image = vision.types.Image(content=content)
		objects = client.object_localization(image=image).localized_object_annotations

	print('Number of objects found: {}'.format(len(objects)))
    	for object_ in objects:
    	# sends out the information of how many faces found in the image to console
    		print('\n{} (confidence: {})'.format(object_.name, object_.score))
		print('Normalized bounding polygon vertices: ')
        # places the location of the detected faces on the image
		for vertex in object_.bounding_poly.normalized_vertices:
			print(' - ({}, {})'.format(vertex.x, vertex.y))

def detect_text(path):
    """Detects text in the file."""
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    string = ''

    for text in texts:
        string+=' ' + text.description
    return string

while(True):
	# Activates the funtion to make the camera output 1080p resolution
	make_1080p()
	# Capture frame-by-frame
	ret, frame = cap.read()
	# File declaring file name for snapshot of the webcam
	file = 'live.png'
	# Name of output image of prossesed image
	out='out.png'
	# Seting the image from the webcam to be processed to a single frame
	cv2.imwrite(file,frame)
	# Call for the main function to process the image and give a output image
	main(file,out,50)
	print(detect_text(file))
	# Setting variable output to be able to be read by opencv to be visable to user
	output=cv2.imread(out)
	# Opens a window of the image that is processed
	cv2.imshow('out',output)

	key = cv2.waitKey(20)
	if key == 27: # exit on ESC
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
