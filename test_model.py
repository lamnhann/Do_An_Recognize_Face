import cv2
import numpy as np
import mysql.connector
#from sklearn.externals import joblib
import joblib
from skimage.feature import hog

# Kết nối tới cơ sở dữ liệu MySQL
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="123456",
  database="thong_tin"
)

# Load mô hình SVM đã được lưu
svm_model = joblib.load('svm_model.pkl')

# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Lặp vô hạn
while(True):
    # Đọc khung hình từ camera
    ret, frame = cap.read()

    # Chuyển đổi khung hình sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Nhận diện khuôn mặt
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Vẽ hình chữ nhật xung quanh khuôn mặt và dự đoán tên
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (100, 100), interpolation = cv2.INTER_AREA)
        # roi = roi_gray.reshape(1, -1)

        def extract_hog_features(X):
            hog_features = []
            for img in X:
                # Kiểm tra xem ảnh đầu vào đã là ảnh xám chưa
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Tính toán đặc trưng HoG cho mỗi hình ảnh

                hog_feature = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                  block_norm='L2-Hys', feature_vector=True)
                # Thêm đặc trưng HoG của hình ảnh vào danh sách hog_features
                hog_features.append(hog_feature)
            # Chuyển đổi danh sách hog_features thành numpy array
            hog_features = np.array(hog_features)
            return hog_features

        roi_hog = extract_hog_features([roi_gray])
        label = svm_model.predict(roi_hog)[0]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Lấy thông tin từ MySQL và hiển thị lên màn hình
        mycursor = mydb.cursor()
        sql = "SELECT * FROM mssv_ten WHERE mssv = %s"
        val = (label,)
        mycursor.execute(sql, val)
        myresult = mycursor.fetchone()
        cv2.putText(frame, myresult[1], (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.putText(frame, myresult[2], (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Mở file text để ghi thông tin điểm danh
        import datetime

        with open('diem_danh.txt', 'a') as f:
            # Lấy thời gian hiện tại
            now = datetime.datetime.now()
            # Ghi thông tin điểm danh và thời gian vào file
            f.write(f"{now:%Y-%m-%d %H:%M:%S} : {myresult[1]} - da diem danh\n")

    # Hiển thị khung hình đã được xử lý
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
