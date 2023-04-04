import cv2
import os
import numpy as np

# Tải dữ liệu hình ảnh
train_dir = 'DATA/train'
test_dir = 'DATA/test'

# train_images = []
# test_images = []
#
# for file in os.listdir(train_dir):
#     if file.endswith('.jpg') or file.endswith('.png'):
#         img = cv2.imread(os.path.join(train_dir, file))
#         train_images.append(img)
#
# for file in os.listdir(test_dir):
#     if file.endswith('.jpg') or file.endswith('.png'):
#         img = cv2.imread(os.path.join(test_dir, file))
#         test_images.append(img)

train_images = []
test_images = []
train_labels = []
test_labels = []

for folder in os.listdir(train_dir):
    folder_path = os.path.join(train_dir, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.jpg') or file.endswith('.png'):
                img = cv2.imread(os.path.join(folder_path, file))
                train_images.append(img)
                train_labels.append(folder)

for folder in os.listdir(test_dir):
    folder_path = os.path.join(test_dir, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.jpg') or file.endswith('.png'):
                img = cv2.imread(os.path.join(folder_path, file))
                test_images.append(img)
                test_labels.append(folder)

# Chuẩn hóa dữ liệu
IMG_HEIGHT = 256
IMG_WIDTH = 256

def resize_images(images, height, width):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, (height, width))
        resized_images.append(resized_img)
    return resized_images

train_images_resized = resize_images(train_images, IMG_HEIGHT, IMG_WIDTH)
test_images_resized = resize_images(test_images, IMG_HEIGHT, IMG_WIDTH)

# Loại bỏ nhiễu
def remove_noise(images):
    denoised_images = []
    for img in images:
        denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        denoised_images.append(denoised_img)
    return denoised_images

train_images_denoised = remove_noise(train_images_resized)
test_images_denoised = remove_noise(test_images_resized)

# Xử lý đối tượng
def detect_objects(images):
    object_images = []
    for img in images:
        object_img = img.copy()
        gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('path/to/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(object_img,(x,y),(x+w,y+h),(255,0,0),2)
        object_images.append(object_img)
    return object_images

train_images_with_objects = detect_objects(train_images_denoised)
test_images_with_objects = detect_objects(test_images_denoised)

# Phân đoạn hình ảnh
def segment_images(images):
    segmented_images = []
    for img in images:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_range = np.array([0, 40, 30])
        upper_range = np.array([43, 255, 255])
        mask = cv2.inRange(img_hsv, lower_range, upper_range)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        img_masked = cv2.bitwise_and(img, img, mask=mask_clean)
        segmented_images.append(img_masked)
    return segmented_images

train_images_segmented = segment_images(train_images_with_objects)
test_images_segmented = segment_images(test_images_with_objects)

#Chuyển đổi hình ảnh sang định dạng grayscale
def convert_to_grayscale(images):
    grayscale_images = []
    for img in images:
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayscale_images.append(grayscale_img)
    return grayscale_images

train_images_grayscale = convert_to_grayscale(train_images_segmented)
test_images_grayscale = convert_to_grayscale(test_images_segmented)

#Tăng cường ảnh (image augmentation)
def image_augmentation(images):
    augmented_images = []
    for img in images:
        # Lật ảnh ngang
        flipped_horizontally = cv2.flip(img, 1)
        augmented_images.append(flipped_horizontally)
        # Lật ảnh dọc
        flipped_vertically = cv2.flip(img, 0)
        augmented_images.append(flipped_vertically)
        # Xoay ảnh
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)
        rotated_img = cv2.warpAffine(img, M, (cols, rows))
        augmented_images.append(rotated_img)
    return augmented_images

train_images_augmented = image_augmentation(train_images_grayscale)

#Chuẩn hoá màu sắc
def normalize_color(images):
    normalized_images = []
    for img in images:
        normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        normalized_images.append(normalized_img)
    return normalized_images

train_images_normalized = normalize_color(train_images_augmented)
test_images_normalized = normalize_color(test_images_grayscale)

# Trích xuất đặc trưng HoG
def extract_hog_features(images):
    hog = cv2.HOGDescriptor()
    hog_features = []
    for img in images:
        hog_feature = hog.compute(img)
        hog_features.append(hog_feature)
    return hog_features

train_features = extract_hog_features(train_images_normalized)
test_features = extract_hog_features(test_images_normalized)

#Xây dưng mô hình CNN


//////////////////////////////////////

# def extract_hog_features(images):
#     features = []
#     for img in images:
#         # resize ảnh về cùng kích thước
#         img_resized = cv2.resize(img, (64, 128))
#         # tính đặc trưng HOG
#         fd = hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
#         features.append(fd)
#     return np.array(features)
#
# train_features = extract_hog_features(train_images_normalized)
# test_features = extract_hog_features(test_images_normalized)


///////////////////////////////////////////////////

#
# from sklearn.model_selection import train_test_split
#
# # X là biến đầu vào, y là biến đầu ra
# X_train, X_test, y_train, y_test = train_test_split(train_images, test_images, test_size=0.2, random_state=42)
#
# # test_size=0.2 cho biết tỷ lệ phần trăm của tập kiểm tra, trong trường hợp này là 20%
# # random_state là một giá trị seed để cố định quá trình chia dữ liệu, giúp cho kết quả của chia dữ liệu này giống nhau mỗi khi chạy lại code







# Trích xuất đặc trưng HoG
def extract_features(images):
    features = []
    for img in images:
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        features.append(hog_features)
    return np.array(features)

train_features = extract_features(train_images_normalized)
test_features = extract_features(test_images_normalized)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra với tỷ lệ 80:20
X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)



# Lưu 300 tấm hình vào thư mục train
    sampleNum = 0
    while sampleNum < 200:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img_name = f"./DATA/train/{folder_name}/img_{forder_text}{sampleNum}.jpg"
            cv2.imwrite(img_name, gray[y:y+h, x:x+w])
            sampleNum += 1
            if sampleNum == 300:
                break
        cv2.imshow("Recognize Face", frame)
        cv2.waitKey(1)

    label_text.configure(text="Lưu dữ liệu thành công!")

    # Lưu 100 tấm hình vào thư mục test
    sampleNum_test = 0
    while sampleNum_test < 100:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img_name = f"./DATA/test/{folder_name}/img_{forder_text}{sampleNum_test}.jpg"
            cv2.imwrite(img_name, gray[y:y + h, x:x + w])
            sampleNum += 1
            if sampleNum == 100:
                break
        cv2.imshow("Recognize Face", frame)
        cv2.waitKey(1)














import cv2
import numpy as np
import mysql.connector
#from sklearn.externals import joblib
import joblib

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
        roi = roi_gray.reshape(1, -1)
        #Viết thêm hàm trích xuút đặc trưng HoG
        label = svm_model.predict(roi)[0]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Lấy thông tin từ MySQL và hiển thị lên màn hình
        mycursor = mydb.cursor()
        sql = "SELECT * FROM mssv_ten WHERE mssv = %s"
        val = (label,)
        mycursor.execute(sql, val)
        myresult = mycursor.fetchone()
        cv2.putText(frame, myresult[1], (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, myresult[2], (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Hiển thị khung hình đã được xử lý
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()







import cv2
import numpy as np
import mysql.connector
# from sklearn.externals import joblib
import joblib

# Load the SVM model
svm = joblib.load('svm_model.pkl')

# Connect to the MySQL database
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="123456",
  database="thong_tin"
)


# Define the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # Extract the face region from the frame
        face = gray[y:y+h, x:x+w]

        # Resize the face region to the same size as the training images
        face = cv2.resize(face, (64, 64))

        # Extract features from the face region using the same method as in training
        features = np.array(face, dtype=np.float32).flatten()

        # Normalize the features to have zero mean and unit variance
        features = (features - np.mean(features)) / np.std(features)
        # Predict the class label of the face using the SVM model
        label = svm.predict([features])[0]

        # Draw a rectangle around the face and display the predicted label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Insert the predicted label into the MySQL database along with the current timestamp
        cursor = mydb.cursor()
        sql = "INSERT INTO faces (label, timestamp) VALUES (%s, NOW())"
        val = (label,)
        cursor.execute(sql, val)
        mydb.commit()

    # Display the frame
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



import cv2
import numpy as np
import mysql.connector
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
        # roi_new = roi_gray.reshape(1, -1)# su dung HoG, viet them ham


        def extract_hog_features(X):
            hog_features = []
            for img in X:
                # Tính toán đặc trưng HoG cho mỗi hình ảnh
                hog_feature = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                  block_norm='L2-Hys')
                # Thêm đặc trưng HoG của hình ảnh vào danh sách hog_features
                hog_features.append(hog_feature)
            # Chuyển đổi danh sách hog_features thành numpy array
            hog_features = np.array(hog_features)
            return hog_features

        roi = extract_hog_features(roi_gray)
        label = svm_model.predict(roi)[0]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # cv2.putText(frame, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Lấy thông tin từ MySQL và hiển thị lên màn hình
        mycursor = mydb.cursor()
        sql = "SELECT * FROM mssv_ten WHERE mssv = %s"
        val = (label,)
        mycursor.execute(sql, val)
        myresult = mycursor.fetchone()
        cv2.putText(frame, myresult[1], (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, myresult[2], (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Hiển thị khung hình đã được xử lý
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
