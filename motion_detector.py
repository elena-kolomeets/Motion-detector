from datetime import datetime
import cv2
import numpy
import pandas


def time_dataframes(time_list):
    df = pandas.DataFrame(columns=["Start", "End"])
    for i in range(0, len(time_list), 2):
        df = df.append({"Start": time_list[i], "End": time_list[i+1]}, ignore_index=True)
    df.to_csv("Times.csv")


def detector():
    """
    Webcam motion detection function.
    Shows the webcam output and draws green rectangles
    around the moving object in the video.
    Works best if the first image of the video is empty
    and static and moving object appears later.
    """
    first_cap = None
    status_list = [0, 0]
    times = []
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        check, frame = cap.read()
        # find a point with no motion
        status = 0
        if check:
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # make the frame blurry for better calculating delta
            gray_current = cv2.GaussianBlur(gray_current, (21, 21), 0)
            # catch the first capture
            if first_cap is None:
                # check if first capture is too dark (then delta is none)
                # and only catch if not dark
                if numpy.mean(gray_current) > 50:
                    first_cap = gray_current
                    continue
            else:
                # calculate difference between first (static) frame and current one
                delta_frame = cv2.absdiff(first_cap, gray_current)
                # make moving area white and the rest black
                thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
                # make white area smooth to detect the contour
                thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
                # finding contours of the moving objects
                (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # count only contours that are big enough
                for contour in cnts:
                    if cv2.contourArea(contour) < 5000:
                        continue
                    # catch the motion
                    status = 1
                    # draw rectangles around moving objects
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                status_list.append(status)

                # catch the time the motion starts
                if status_list[-2] == 0 and status_list[-1] == 1:
                    times.append(datetime.now())
                # catch the time the motion stops
                if status_list[-2] == 1 and status_list[-1] == 0:
                    times.append(datetime.now())

            cv2.imshow("Capture", frame)
            if cv2.waitKey(1) == ord("q"):
                if status == 1:
                    times.append(datetime.now())
                break
    cap.release()
    cv2.destroyAllWindows()
    return times


if __name__ == '__main__':
    movement_times = detector()
    time_dataframes(movement_times)
