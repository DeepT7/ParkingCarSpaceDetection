import cv2 

video_path = 'videos/parking_lot_1.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    suc, frame = cap.read()
    cv2.imwrite('data/frame.png', frame)
    cv2.waitKey(1)
    break