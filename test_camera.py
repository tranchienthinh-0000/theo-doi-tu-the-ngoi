import cv2

for i in range(5):  # Thử ID 0 đến 4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera hoạt động với ID {i}")
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Camera ID {i}", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"ID {i} không hoạt động")