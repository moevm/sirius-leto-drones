import cv2

def blur_cv(data, k_size=5):
    kernel_size = (k_size, k_size)  
    return cv2.blur(data, kernel_size)

def blur_gauss(data, k_size=5, sigmaX=0, sigmaY=0):
    kernel_size = (k_size, k_size)
    return cv2.GaussianBlur(data, kernel_size, sigmaX, sigmaY)
