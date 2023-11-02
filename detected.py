import cv2 as cv
import numpy as np
import statistics
import pytesseract
from PIL import Image

def center_images(image_data):
    height, width = image_data.shape

    leftmost_point = None
    topmost_point = None
    rightmost_point = None
    bottommost_point = None

    # 遍历图像数组
    for y in range(height):
        for x in range(width):
            pixel_value = image_data[y, x]
            
            if pixel_value == 0:
                if leftmost_point is None or x < leftmost_point[0]:
                    leftmost_point = (x, y)
                
                if topmost_point is None or y < topmost_point[1]:
                    topmost_point = (x, y)
                
                if rightmost_point is None or x > rightmost_point[0]:
                    rightmost_point = (x, y)
                
                if bottommost_point is None or y > bottommost_point[1]:
                    bottommost_point = (x, y)

    # 计算矩形的宽度和高度
    rect_width = rightmost_point[0] - leftmost_point[0]
    rect_height = bottommost_point[1] - topmost_point[1]
    print('height, width: ', rect_height, rect_width)

    # 计算图像中心的坐标
    center_x = width // 2
    center_y = height // 2

    # 创建一个新的图像，大小与原图相同
    new_image = np.full((height, width), 255, dtype=np.uint8)

    # 计算矩形内所有像素点应该移动的偏移量
    offset_x = center_x - (leftmost_point[0] + rightmost_point[0]) // 2
    offset_y = center_y - (topmost_point[1] + bottommost_point[1]) // 2

    # 遍历矩形内的像素点，并将其移动到图像中心
    for y in range(topmost_point[1], bottommost_point[1] + 1):
        for x in range(leftmost_point[0], rightmost_point[0] + 1):
            # 计算像素点在新图像中的位置
            new_x = x + offset_x
            new_y = y + offset_y
            
            if (new_x < 0 or new_x >= width or new_y < 0 or new_y >= height):
                continue

            new_image[new_y, new_x] = image_data[y, x]
    return new_image

# 创建一个窗口
trackbar_window_name = "trackbar"
cv.namedWindow(trackbar_window_name)

def addTrackbar(name, start, max, callback):
    cv.createTrackbar(name, trackbar_window_name, start, max, callback)

def canny_threshold_min_cb(value):
    global canny_threshold_min
    canny_threshold_min = value
def canny_threshold_max_cb(value):
    global canny_threshold_max
    canny_threshold_max = value

canny_threshold_min = 180
canny_threshold_max = 190
addTrackbar("canny_min", canny_threshold_min, 255, canny_threshold_min_cb)
addTrackbar("canny_max", canny_threshold_max, 255, canny_threshold_max_cb)

img = cv.imread("board.png")
img = cv.GaussianBlur(img, (3, 3), 0)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# BGR
print('color:', len(img[0]), 'gray:', len(gray[0]))

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
sharpened = cv.filter2D(gray, -1, kernel)

edges = cv.Canny(sharpened,
                 threshold1=canny_threshold_min,
                 threshold2=canny_threshold_max)

# 圆检测
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=20,
                         param1=180, param2=30, minRadius=10, maxRadius=30)

# print(template)
# cv.imshow('template', template)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    circle_r = statistics.median([r for (_, _, r) in circles]).astype("int")
    # print(circle_r)
    for index in range(32):
        template = cv.imread('test' + str(index + 1) + '.png', cv.IMREAD_GRAYSCALE)

        roi_array = []
        i = 1
        for (x, y, r) in circles:
            # print(x, y, r, circle_r)
            cv.circle(img, (x, y), circle_r + 5, (255, 255, 255), 20)
            roi = img[y - circle_r:y + circle_r, x - circle_r:x + circle_r]
            roi_copy = roi.copy()
            # roi_array.append(roi.copy())
            # cv.circle(img, (x, y), circle_r, (0, 255, 0), 2)
            roi_gray = cv.cvtColor(roi_copy, cv.COLOR_BGR2GRAY)

            sharpened = cv.filter2D(roi_gray, -1, kernel)
            denoised = cv.medianBlur(sharpened, 3)
            roi_threshold = cv.threshold(denoised, 20, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
            roi_threshold = cv.dilate(roi_threshold, np.ones((3, 3), np.uint8), iterations=1)
            roi_threshold = cv.erode(roi_threshold, np.ones((3, 3), np.uint8), iterations=1)
            # roi_threshold = cv.GaussianBlur(roi_threshold, (3, 3), 0)
            # roi_threshold = cv.threshold(roi_threshold, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
            roi_threshold = center_images(roi_threshold)

            result = cv.matchTemplate(roi_threshold, template, cv.TM_CCORR_NORMED)
            print('match score: ', np.max(result))
            t_threshold = 0.91
            locations = np.where(result >= t_threshold)
            locations = list(zip(*locations[::-1]))
            for loc in locations:
                top_left = loc
                bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
                cv.rectangle(roi_threshold, top_left, bottom_right, (0, 255, 0), 2)
            roi_array.append(roi_threshold)

            # Image.fromarray(roi_threshold).save('test' + str(i) + '.png')
            # i = i + 1

            # text = pytesseract.image_to_string(Image.fromarray(roi_threshold), lang='chi_sim')
            # print(text)
        # print(roi_array)
        roi_img = np.hstack(tuple(roi_array))
        cv.imshow('ROI' + str(index + 1), roi_img)

# image = Image.open('test1.png')
# text = pytesseract.image_to_string(image, lang='chi_sim') 
# print(text)

# 直线检测
# lines = cv.HoughLines(edges, 1, np.pi / 180, 400)
# if lines is not None:
#     print(len(lines))
#     for line in lines:
#         rho, theta = line[0]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 角点检测
# corners = cv.goodFeaturesToTrack(gray, 100, 0.01, 10)
# if corners is not None:
#     corners = np.intp(corners)
#     for corner in corners:
#         x, y = corner.ravel()
#         cv.circle(img, (x, y), 3, (255, 0, 0), -1)

# 设置棋盘格尺寸
# gray = np.float32(gray)
# dst = cv.cornerHarris(gray, 3, 3, 0.1)
# dst = cv.dilate(dst, None)
# img[dst>0.01*dst.max()] = [255, 0, 0]
# cv.imshow("result_img", img)
cv.waitKey(0)
cv.destroyAllWindows()

# while True:
#     edges = cv.Canny(sharpened,
#                      threshold1=canny_threshold_min,
#                      threshold2=canny_threshold_max)
#
#     circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=100,
#                              param1=180, param2=30, minRadius=10, maxRadius=50)
#     circles = np.round(circles[0, :]).astype("int")
#     for (x, y, r) in circles:
#         cv.circle(gray, (x, y), r, (0, 255, 0), 2)
#
#     # res = np.hstack((img, img))
#     cv.imshow('result_img', gray)
#     key = cv.waitKey(1) & 0xFF
#     if key == ord('q'):
#         cv.destroyAllWindows()
#         break

# sharpened = cv.filter2D(sharpened, -1, kernel)
# res = np.hstack((gray, denoised))
# cv.imshow('result_img', res)

# def on_trackbar(value):
#     _, threshold = cv.threshold(sharpened, value, 255, cv.THRESH_BINARY)
#     denoised = cv.medianBlur(threshold, 3)
#     combined_img = np.hstack((sharpened, threshold, denoised))
#     cv.imshow("result_img", combined_img)
#
# cv.createTrackbar("Trackbar Name", "Trackbar Window", threshold_value, 255, on_trackbar)
#
# while True:
#     key = cv.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
