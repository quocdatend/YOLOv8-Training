import os
import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from collections import defaultdict

def get_selected_option():
    is_Save = False
    selected_option = var.get()
    if selected_option == 1:
        # Load the YOLOv8 model
        model = YOLO('Yolov8n.pt')
    elif selected_option == 2:
        # Load the YOLOv8 model
        model = YOLO('Yolov8n-seg.pt')
    elif selected_option == 3:
        # Load the YOLOv8 model
        model = YOLO('Yolov8n-cls.pt')
    elif selected_option == 4:
        # Load the YOLOv8 model
        model = YOLO('Yolov8n-pose.pt')
    elif selected_option == 5:
        # Load the YOLOv8 model
        model = YOLO('Yolov8n.pt')
    else:
        model = YOLO('Yolov8n.pt')

    if var_yes_no.get() == 1:
        is_Save = True

    # Create a VideoCapture object to capture video from the default camera (usually the webcam)
    cap = cv2.VideoCapture(label_File.cget("text") != "" and label_File.cget("text") or 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1050)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

    track_history = defaultdict(lambda: [])

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera or video or image")
    else:
        # Read and display frames from the camera
        while True:
            ret, frame = cap.read()  # Capture frame-by-frame

            # check ret false when i chose image input
            if not ret:
                break

            if selected_option == 5:
                # Run YOLOv8 tracking on the frame, conf, save, exist_ok, persisting tracks between frames
                results = model.track(frame, conf=float(input_conf.get() == "" and 0.4 or input_conf.get()), save=is_Save, exist_ok=True, persist=True)
            
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                # Run YOLOv8 inference on the frame
                results = model(frame, conf=float(input_conf.get() == "" and 0.4 or input_conf.get()), save=is_Save, exist_ok=True)
                
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            if selected_option == 5:
                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)


            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference(Press enter to exit!)", annotated_frame)

            # cv2.imshow("MASK", mask)

            # Break the loop when 'Enter' is pressed or ret is false
            if cv2.waitKey(1) & 0xFF == 13:
                break

    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()

# func open folder from value input or show error if path is false
def open_folder():
    folder_path = (input_path.get() == "" and "C:" or input_path.get()) 
    if os.path.exists(folder_path):
        os.startfile(folder_path)
    else:
        tk.messagebox.showerror("Lỗi", "Thư mục không tồn tại")

#  func open file and paste to label
def choose_File():
    selected_File = filedialog.askopenfilename()
    if selected_File:
        label_File.config(text=f"{selected_File}")

# clear value in label
def clear_file():
    label_File.config(text="")


root = tk.Tk(className="YOLOv8")

window_width = 500
window_height = 420

# Lấy kích thước màn hình
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Tính toán vị trí xuất hiện
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2

# Cài đặt kích thước và vị trí xuất hiện
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

var = tk.IntVar()
var_yes_no = tk.IntVar()


title_type = tk.Label(root, text="Chọn dữ liệu đầu vào(Mặc định là camera)!")
open_default_folder_button = tk.Button(root, text="Open", command=choose_File)
clear_file_button = tk.Button(root, text="Clear", command=clear_file)
label_File = tk.Label(root, text="")
label = tk.Label(root, text="Chọn một trong ba tùy chọn(Mặc định là Detection):")
radio_button_1 = tk.Radiobutton(root, text="Detection", variable=var, value=1)
radio_button_2 = tk.Radiobutton(root, text="Segmentation", variable=var, value=2)
radio_button_3 = tk.Radiobutton(root, text="Classification", variable=var, value=3)
radio_button_4 = tk.Radiobutton(root, text="Pose", variable=var, value=4)
radio_button_5 = tk.Radiobutton(root, text="Track", variable=var, value=5)
title_conf = tk.Label(root, text="Nhập độ chính xác cho model(0->1)(Mặc định 0.4):")
input_conf = tk.Entry(root)
label_yes_no = tk.Label(root, text="Lưu dữ liệu hay không!")
check_button_yes_no = tk.Checkbutton(root, text="Save", variable=var_yes_no, onvalue=1, offvalue=0)
title_path = tk.Label(root, text="Nhập Path Data:")
input_path = tk.Entry(root)
open_folder_button = tk.Button(root, text="Mở thư mục(Mặc định C:)", command=open_folder)
select_button = tk.Button(root, text="Run", command=get_selected_option)


title_type.pack()
open_default_folder_button.pack()
label_File.pack()
clear_file_button.pack()
label.pack()
radio_button_1.pack()
radio_button_2.pack()
radio_button_3.pack()
radio_button_4.pack()
radio_button_5.pack()
title_conf.pack()
input_conf.pack()
label_yes_no.pack()
check_button_yes_no.pack()
title_path.pack()
input_path.pack()
open_folder_button.pack()
select_button.pack()

root.mainloop()
