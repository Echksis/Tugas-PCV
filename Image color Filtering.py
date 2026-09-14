import cv2

# Read Image
image = cv2.imread("earth.jpg")


cv2.waitKey(0)
cv2.destroyAllWindows()


[h, w, c] = image.shape

for i in range(h):
    for j in range(w):
        image[i, j, 1] = 0
        image[i, j, 2] = 0
        
cv2.imshow("Filtered Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()