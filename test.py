import cv2

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
pic = cv2.imread('java.jpg')
fgmask = fgbg.apply(pic)
cv2.imshow(fgmask)
## Further improvement :- Semantic Map Based Style Transfer