import numpy as np
import cv2
import apriltag as at


def detect_apriltags(input_img: np.ndarray) -> np.ndarray:
    """
    :param ndarray input_img: represents RGB opencv image
    :returns ndarray: input_img copy with apriltags detected and marked
    """
    result_img = input_img.copy()
    grayscale_input_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY)
    options = at.DetectorOptions(families="tag36h10")
    detector = at.Detector(options)
    detected_ats = detector.detect(grayscale_input_img)
    print(f"[INFO] drone detected {len(detected_ats)} apriltags")
    for detection in detected_ats:
        corners = np.array(detection.corners, dtype=np.int32)
        cv2.polylines(result_img, [corners], True, (0, 255, 0), 2)
        center = np.array(detection.center, dtype=np.int32)
        cv2.circle(result_img, tuple(center), 2, (0, 0, 255), -1)
        tag_id = detection.tag_id
        cv2.putText(result_img, str(tag_id), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return result_img


def main():
    example = cv2.imread('images/example2.png')
    detect_apriltags(example)
    cv2.imwrite('images/example2_detected.png', example)


if __name__ == '__main__':
    main()