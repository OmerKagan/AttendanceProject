import cv2
import numpy as np
import face_recognition

# imgElon = face_recognition.load_image_file("ImagesBasic/Elon Musk.jpg")
# imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgSelcuk = face_recognition.load_image_file("ImagesBasic/Selcuk Bayraktar.jpg")
imgSelcuk = cv2.cvtColor(imgSelcuk, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file("ImagesBasic/SelcukB Test.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgSelcuk)[0] #1st element of the locations list(top, right, bottom, left)
encodeSelcuk = face_recognition.face_encodings(imgSelcuk)[0]
cv2.rectangle(imgSelcuk, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeSelcuk], encodeTest)
faceDis = face_recognition.face_distance([encodeSelcuk], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

cv2.imshow("Selcuk Bayraktar", imgSelcuk)
cv2.imshow("Selcuk Test", imgTest)
cv2.waitKey(0)