import cv2

from cv2.aruco import CharucoBoard
from cv2 import aruco


board = CharucoBoard(size=(5, 7), squareLength=0.03, markerLength=0.015, dictionary=aruco.getPredefinedDictionary(aruco.DICT_4X4_50))

img = board.generateImage(outSize=(1366, 768), marginSize=10, borderBits=1)

cv2.imshow(winname="res", mat=img)

cv2.waitKey(delay=0)
