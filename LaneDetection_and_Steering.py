import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate deviation from a straight line
def calculate_deviation(left_x, right_x, image_width):
    lane_center = (left_x + right_x) // 2
    deviation = lane_center - (image_width // 2)
    return deviation


# Function to adjust steering angle based on deviation
def adjust_steering(deviation):
    steering_angle = deviation * 0.1  # Adjust this factor as needed
    return steering_angle


# Initialize video capture
# vidcap = cv2.VideoCapture("resources\LaneVideo.mp4")
vidcap = cv2.VideoCapture("C:/Users/Dell/Videos/Captures/Roads.mp4")




steering_wheel = cv2.imread("resources/steering_wheel.png", cv2.IMREAD_UNCHANGED)
wheel_center = (steering_wheel.shape[1] // 2, steering_wheel.shape[0] // 2)
wheel_radius = min(wheel_center)

while 1:
    success, image = vidcap.read()
    frame = cv2.resize(image, (640, 480))

    # Choosing points for perspective transformation
    tl = (180, 380)
    bl = (80, 480)

    tr = (440, 380)
    br = (520, 480)

    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)

    # Applying bird's eye view / perspective transformation
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    # Image Thresholding in HSV format
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

    white_lower = np.array([0, 0, 100])
    white_upper = np.array([255, 220, 220])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    white_mask = cv2.inRange(hsv_transformed_frame, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsv_transformed_frame, yellow_lower, yellow_upper)

    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Histogram
    histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
    midpoint = np.int32(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    plt.plot(histogram)
    plt.show()
    # Sliding Window
    y = 472
    lx = []
    rx = []
    msk = mask.copy()

    while y > 0:
        # Left threshold
        img = mask[y - 40:y, left_base - 50:left_base + 50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cx_global = left_base - 50 + cx
                lx.append(cx_global)
                left_base = cx_global

        # Right threshold
        img = mask[y - 40:y, right_base - 50:right_base + 50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cx_global = right_base - 50 + cx
                rx.append(cx_global)
                right_base = cx_global

        cv2.rectangle(msk, (left_base - 50, y), (left_base + 50, y - 40), (255, 255, 255), 1)
        cv2.rectangle(msk, (right_base - 50, y), (right_base + 50, y - 40), (255, 255, 255), 1)
        y -= 40

    # Calculate deviation from a straight line if lanes detected
    if lx and rx:
        deviation = calculate_deviation(min(lx), max(rx), mask.shape[1])
    else:
        deviation = 0

    # Adjust steering angle based on deviation
    if deviation > abs(10):
        steering_angle = 5 * adjust_steering(deviation)
        print("Steering Angle:", steering_angle)

        # Rotate steering wheel image
        rotation_matrix = cv2.getRotationMatrix2D(wheel_center, -steering_angle, 1)
        rotated_wheel = cv2.warpAffine(steering_wheel, rotation_matrix,
                                       (steering_wheel.shape[1], steering_wheel.shape[0]))
        cv2.imshow("Original", frame)
        cv2.imshow("Bird's Eye View", hsv_transformed_frame)
        cv2.imshow("Lane Detection - Image Thresholding", mask)
        cv2.imshow("Lane Detection - Sliding Windows", msk)
        cv2.imshow("Steering Wheel", rotated_wheel)

    else:
        cv2.imshow("Original", frame)
        cv2.imshow("Bird's Eye View", hsv_transformed_frame)
        cv2.imshow("Lane Detection - Image Thresholding", mask)
        cv2.imshow("Lane Detection - Sliding Windows", msk)
        cv2.imshow("Steering Wheel", steering_wheel)

    if cv2.waitKey(10) == ord('q'):
        break

