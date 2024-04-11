import cv2

def blur_cv(data):
    kernel_size = (5, 5)  
    return cv2.blur(data, kernel_size)

def blur_gauss(data, sigmaX=0, sigmaY=0):
    kernel_size = (5, 5)
    return cv2.GaussianBlur(data, kernel_size, sigmaX, sigmaY)
