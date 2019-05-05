import numpy as np
import cv2 as cv

def generate_trimap(alpha):
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    # fg = cv.erode(fg, kernel, iterations=np.random.randint(1, 3))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)

for imgNum in range(1,631,1):
	alpha_img=cv.imread('../dataset/train2/alpha/%05d.png'%imgNum)
	trimap_img=generate_trimap(alpha_img)
	cv.imwrite('../dataset/train2/trimap/%05d.png'%imgNum,trimap_img)