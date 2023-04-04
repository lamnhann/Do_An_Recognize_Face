import cv2
import tkinter as tk
from PIL import ImageTk, Image
import mysql.connector
import os
import sys

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="123456",
  database="thong_tin"
)

# Tạo đối tượng cursor để thực hiện các truy vấn
mycursor = mydb.cursor()

# Tạo bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Hàm xử lý sự kiện khi người dùng nhấn nút
def save_to_database():
    # Lấy giá trị từ ô văn bản
    text_mssv = text_input1.get()
    text_ten = text_input2.get()

    # Thực hiện truy vấn để lưu giá trị vào cơ sở dữ liệu
    sql = "INSERT INTO mssv_ten (mssv, hoten) VALUES (%s, %s)"
    val = (text_mssv, text_ten)
    mycursor.execute(sql, val)

    # Lưu thay đổi vào cơ sở dữ liệu
    mydb.commit()

    # Lấy tên thư mục từ ô văn bản
    folder_name = text_input1.get()
    forder_text = text_input2.get()

    # Tạo thư mục mới
    os.mkdir("./DATA/train/" + folder_name)
    os.mkdir("./DATA/test/" + folder_name)

    # Lưu 300 tấm hình vào thư mục train và 100 tấm hình vào thư mục test
    sampleNum_train, sampleNum_test = 0, 0
    while sampleNum_train < 300 or sampleNum_test < 100:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if sampleNum_train < 300:
                img_name_train = f"./DATA/train/{folder_name}/img_{forder_text}{sampleNum_train}.jpg"
                cv2.imwrite(img_name_train, gray[y:y + h, x:x + w])
                sampleNum_train += 1
            if sampleNum_test < 100:
                img_name_test = f"./DATA/test/{folder_name}/img_{forder_text}{sampleNum_test}.jpg"
                cv2.imwrite(img_name_test, gray[y:y + h, x:x + w])
                sampleNum_test += 1
        cv2.imshow("Recognize Face", frame)
        cv2.waitKey(1)
    label_text.configure(text="Lưu dữ liệu thành công!")
    cv2.destroyAllWindows()
    sys.exit()


def show_frame():
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label_camera.imgtk = imgtk
    label_camera.configure(image=imgtk)
    label_camera.after(10, show_frame)

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Recognize Face")

# Tạo các đối tượng giao diện
button = tk.Button(root, text="Lưu dữ liệu", command=save_to_database)
label_mssv = tk.Label(root, text = "MSSV")
text_input1 = tk.Entry(root, width= 50)
label_name = tk.Label(root, text="Họ và tên")
text_input2 = tk.Entry(root, width= 50)
label_text = tk.Label(root, text="")
label_camera = tk.Label(root)

# Hiển thị các đối tượng giao diện

label_mssv.pack()
text_input1.pack()

label_name.pack()
text_input2.pack()

button.pack()

label_text.pack()
label_camera.pack()

# Khởi tạo camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Bắt đầu hiển thị camera
show_frame()

# Khởi chạy vòng lặp chính
root.mainloop()

# Giải phóng camera
cap.release()
