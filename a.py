import cv2


def tint_merah(frame):
    hasil = frame.copy()

    hasil[:, :, 0] = 255  # Blue 
    hasil[:, :, 1] = 0  # Green 

    return hasil


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)  

        frame_merah = tint_merah(frame)

        cv2.imshow("Output", frame_merah)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()