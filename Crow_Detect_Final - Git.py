import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def detection(frame, model):
    frame = [frame]
    results = model(frame)

    labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, coordinates

def plot_boxes(results, frame, classes):
    labels, cord = results
    l = len(labels)

    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"Number of detections: {l}")

    data = []

    for i in range(l):
        row = cord[i]
        if row[4] >= 0.75: # Confidence threshold
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            name = classes[int(labels[i])]

            if name == 'crow':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Bounding Box
                cv2.rectangle(frame, (x1, y1-15), (x2, y1), (0, 0, 0), -1) # Top text label
                cv2.rectangle(frame, (x1+10, y1+10), (x1, y1), (0, 0, 0), -1) # Top left corner coordinates box
                cv2.rectangle(frame, (x1-130, y1+30), (x1, y1), (0, 0, 0), -1) # Top left corner label
                cv2.rectangle(frame, (x2-10, y2-10), (x2, y2), (0, 0, 0), -1) # Bottom right corner coordinates box
                cv2.rectangle(frame, (x2+130, y2-30), (x2, y2), (0, 0, 0), -1) # Bottom right corner label
                cx, cy = ((x1 + x2)//2, (y1 + y2)//2) # Centroid coordinates formula
                print(f"Centroid: {cx}, {cy}")
                centroid = cx, cy
                x1y1 = x1, y1
                x2y2 = x2, y2
                confidence = {round(float(row[4]), 2)}
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1) # Centroid coordinates

                cv2.putText(frame, f"{cx, cy}", (cx+15, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2) # Centroid coordinates label
                cv2.putText(frame, f"{x1, y1}", (x1-115, y1+20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2) # Top left coordinates label
                cv2.putText(frame, f"{x2, y2}", (x2+10, y2-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2) # Bottom right coordinates label
                cv2.putText(frame, name + f" {round(float(row[4]), 2)}", (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)) # Class name label('crow')
                print(f"x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}")
                print(f"Confidence:{round(float(row[4]), 2)}")
                #print(centroid, x1y1, x2y2)

            elif name == 'bird':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Bounding Box
                cv2.rectangle(frame, (x1, y1-15), (x2, y1), (0, 0, 0), -1) # Top text label
                cv2.rectangle(frame, (x1+10, y1+10), (x1, y1), (0, 0, 0), -1) # Top left corner coordinates box
                cv2.rectangle(frame, (x1-130, y1+30), (x1, y1), (0, 0, 0), -1) # Top left corner label
                cv2.rectangle(frame, (x2-10, y2-10), (x2, y2), (0, 0, 0), -1) # Bottom right corner coordinates box
                cv2.rectangle(frame, (x2+130, y2-30), (x2, y2), (0, 0, 0), -1) # Bottom right corner label
                cx, cy = ((x1 + x2)//2, (y1 + y2)//2) # Centroid coordinates formula
                print(f"Centroid: {cx}, {cy}")
                centroid = cx, cy
                x1y1 = x1, y1
                x2y2 = x2, y2
                confidence = {round(float(row[4]), 2)}
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1) # Centroid coordinates

                cv2.putText(frame, f"{cx, cy}", (cx+15, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2) # Centroid coordinates label
                cv2.putText(frame, f"{x1, y1}", (x1-115, y1+20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2) # Top left coordinates label
                cv2.putText(frame, f"{x2, y2}", (x2+10, y2-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2) # Bottom right coordinates label
                cv2.putText(frame, name + f" {round(float(row[4]), 2)}", (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)) # Class name label('crow')
                print(f"x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}")
                print(f"Confidence:{round(float(row[4]), 2)}")
                #print(centroid, x1y1, x2y2)

                data.append([x1y1, x2y2, centroid, confidence])

    df = pd.DataFrame(data, columns=["x1y1(TopLeft)", "x2y2(BottomRight)", "Centroid", "Confidence"])
    df.style
    #df.to_csv("dataframe_1.csv", index=False)

    return frame

model_local = torch.hub.load('/path/to/main/weights/folder', 'custom', source='local', path='path/to/weights', force_reload=True)
model_git = torch.hub.load('ultralytics/yolov5', 'custom', source='github', path='yolov5s.pt', force_reload=True)

classes_local = model_local.names
classes_git = model_git.names

def main_l(img=None, vid=None, vid_save=None):

    if vid != None:
        cap = cv2.VideoCapture(vid)

        if vid_save:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            save = cv2.VideoWriter(vid_name + ".avi", fourcc, 20.0, size)

        #cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detection(frame, model=model_local)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = plot_boxes(results, frame, classes=classes_local)

                cv2.imshow("Main_camera", frame)

                if vid_save:
                    save.write(frame)

                if cv2.waitKey(1) == ord("q"):
                    break
        
        if vid_save:
            save.release()
            cv2.destroyAllWindows()
        cap.release()
        cv2.destroyAllWindows()

    if img != None:
        frame = cv2.imread(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detection(frame, model=model_local)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame, classes=classes_local)

        cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)

        while True:
            cv2.imshow("Inference", frame)

            if cv2.waitKey(1) == ord("q"):
                while True:
                    ask = input("Would you like to save this image (y/n). ")
                    if ask == "y":
                        name = input("What name would you like to save it as: ")
                        if name:
                            name = name + ".jpg"
                            cv2.imwrite(name, frame)
                            quit()
                    elif ask == "n":
                        quit()
                    else:
                        print("Invalid. (y = yes, n = no)")
                        continue

def main_g(img=None, vid=None, vid_save=None):

    if vid != None:
        cap = cv2.VideoCapture(vid)

        if vid_save:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            save = cv2.VideoWriter(vid_name + ".avi", fourcc, 20.0, size)

        #cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detection(frame, model=model_git)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = plot_boxes(results, frame, classes=classes_git)

                cv2.imshow("Main_camera", frame)

                if vid_save:
                    save.write(frame)

                if cv2.waitKey(1) == ord("q"):
                    break
        
        if vid_save:
            save.release()
            cv2.destroyAllWindows()
        cap.release()
        cv2.destroyAllWindows()

    if img != None:
        frame = cv2.imread(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detection(frame, model=model_git)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame, classes=classes_git)

        cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)

        while True:
            cv2.imshow("Inference", frame)

            if cv2.waitKey(1) == ord("q"):
                while True:
                    ask = input("Would you like to save this image (y/n). ")
                    if ask == "y":
                        name = input("What name would you like to save it as: ")
                        if name:
                            name = name + ".jpg"
                            cv2.imwrite(name, frame)
                            quit()
                    elif ask == "n":
                        quit()
                    else:
                        print("Invalid. (y = yes, n = no)")
                        continue

while True:
    val = input("Which source would you like to use? (Input 0, 1, or 2) 0=Webcam, 1=Video, 2=Image (q to quit at any time(Must press on video/img, when applicable)): ")
    #print(f"You chose ({int(val)})")
    if val == "q":
        break

    

    try:
        option = int(val)
        #print("This is a number")
        if option == 0:
            while True:
                mode = input("Would you like to use your local yolov5 model or ultralytics github model(detects any bird NOT only crows)? (Input local or git) ")
                if mode == "local":
                    main_l(vid=0)
                    if main_l:
                        break
                elif mode == "git":
                    main_g(vid=0)
                    if main_g:
                        break
                elif mode == "q":
                    break
                else:
                    print("Invalid input. Please try again or press q to quit. ")
                    continue


        elif option == 1:
            while True:
                vid_file = input("Enter video file path: ")
                vid_path = Path(vid_file)
                if vid_path.is_file():
                    vid_ask = input("Would you like to save this video (y/n). ")
                    if vid_ask == "y":
                        while True:
                            vid_name = input("What name would you like to save it as: ")
                            if vid_name:
                                mode = input("Would you like to use your local yolov5 model or ultralytics github model(detects any bird NOT only crows)? (Input local or git) ")
                                vid_path = str(vid_path)
                                if mode == "local":
                                    #print(type(vid_path))
                                    main_l(vid=vid_path, vid_save=vid_name)
                                    if main_l:
                                        break
                                elif mode == "git":
                                    main_g(vid=vid_path, vid_save=vid_name)
                                    if main_g:
                                        break
                                elif mode == "q":
                                    break
                                else:
                                    print("Invalid input. Please try again or press q to quit. ")
                                    continue
                                    
                    elif vid_ask == "n":
                        while True:
                            mode = input("Would you like to use your local yolov5 model or ultralytics github model(detects any bird NOT only crows)? (Input local or git) ")
                            vid_path = str(vid_path)
                            if mode == "local":
                                main_l(vid=vid_path, vid_save=None)
                                if main_l:
                                    break
                            elif mode == "git":
                                main_g(vid=vid_path, vid_save=None)
                                if main_g:
                                    break
                            elif mode == "q":
                                break
                            else:
                                print("Invalid input.")
                elif vid_file == "q":
                    break
                else:
                    print("File not found. Please use valid file path.")
                    continue

        elif option == 2:
            while True:
                img_file = input("Enter image file path: ")
                img_path = Path(img_file)
                if img_path.is_file():
                    while True:
                        mode = input("Would you like to use your local yolov5 model or ultralytics github model(detects any bird NOT only crows)? (Input local or git) ")
                        img_path = str(img_path)
                        #print(type(img_path))
                        if mode == "local":
                            main_l(img=img_path)
                            if main_l:
                                break
                        elif mode == "git":
                            main_g(img=img_path)
                            if main_g:
                                break
                        elif mode == "q":
                            break
                        else:
                            print("Invalid input. Please try again or press q to quit. ")
                            continue
                elif img_file == "q":
                    break

                else:
                    print("File not found. Please use valid file path.")
                    continue

        elif option == "q":
            break
        else:
            print("Please select number from 0 - 2.")
    except ValueError:
        print("Please enter a number.")
        continue