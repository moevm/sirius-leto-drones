import numpy as np
import cv2
import apriltag as at


def detect_apriltags_with_blur(input_img: np.ndarray, square_flag: bool) -> np.ndarray:
    """
    :param ndarray input_img: represents RGB opencv image
    :param bool square_flag: flag to find area of tag 
    
    if square_flag:
        :returns ndarray: input_img copy with apriltags detected and marked
        :returns center - center of tag
        :returns square - value of tag square    
    else:
        :returns ndarray: input_img copy with apriltags detected and marked
        :returns center - center of tag
    """
    result_img = input_img.copy()
    grayscale_input_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY)
    grayscale_input_img= cv2.GaussianBlur(grayscale_input_img,(3,3),0 )
    r, grayscale_input_img = cv2.threshold(grayscale_input_img, 80, 255, 0)

    options = at.DetectorOptions(families="tag36h10")
    detector = at.Detector(options)
    detected_ats = detector.detect(grayscale_input_img)
    print(f"[INFO] drone detected {len(detected_ats)} apriltags")
    center = None
    square = None
    centers = []
    id_tags = []
    squares = []
    for detection in detected_ats:
        corners = np.array(detection.corners, dtype=np.int32)
        cv2.polylines(grayscale_input_img, [corners], True, (0, 255, 0), 2)
        center = np.array(detection.center, dtype=np.int32)
        centers.append(center)
        cv2.circle(grayscale_input_img, tuple(center), 2, (0, 0, 255), -1)
        tag_id = detection.tag_id
        id_tags.append(tag_id)
        cv2.putText(grayscale_input_img, str(tag_id), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        if square_flag:
            square = cv2.contourArea(corners)
            squares.append(square)
    if square_flag:
        return grayscale_input_img, centers, squares, id_tags
    else:
        return grayscale_input_img, centers



def detect_apriltags(input_img: np.ndarray, square_flag: bool) -> np.ndarray:
    """
    :param ndarray input_img: represents RGB opencv image
    :param bool square_flag: flag to find area of tag 
    
    if square_flag:
        :returns ndarray: input_img copy with apriltags detected and marked
        :returns center - center of tag
        :returns square - value of tag square    
    else:
        :returns ndarray: input_img copy with apriltags detected and marked
        :returns center - center of tag
    """
    result_img = input_img.copy()
    grayscale_input_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY)

    options = at.DetectorOptions(families="tag36h10")
    detector = at.Detector(options)
    detected_ats = detector.detect(grayscale_input_img)
    print(f"[INFO] drone detected {len(detected_ats)} apriltags")
    center = None
    square = None
    centers = []
    id_tags = []
    squares = []
    for detection in detected_ats:
        corners = np.array(detection.corners, dtype=np.int32)
        cv2.polylines(result_img, [corners], True, (0, 255, 0), 2)
        center = np.array(detection.center, dtype=np.int32)
        centers.append(center)
        cv2.circle(result_img, tuple(center), 2, (0, 0, 255), -1)
        tag_id = detection.tag_id
        id_tags.append(tag_id)
        cv2.putText(result_img, str(tag_id), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        if square_flag:
            square = cv2.contourArea(corners)
            squares.append(square)
    if square_flag:
        return result_img, centers, squares, id_tags
    else:
        return result_img, centers


def main():
    example = cv2.imread('images/example2.png')
    detect_apriltags(example)
    cv2.imwrite('images/example2_detected.png', example)


if __name__ == '__main__':
    main()
