import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn import svm

# Đường dẫn tới thư mục chứa dữ liệu train và test
data_train_path = 'DATA/train'
data_test_path = 'DATA/test'

# Khởi tạo danh sách tên các nhãn (tên thư mục) trong thư mục train
labels_train = os.listdir(data_train_path)

# Khởi tạo danh sách tên các nhãn (tên thư mục) trong thư mục test
labels_test = os.listdir(data_test_path)

# Khởi tạo 2 danh sách chứa dữ liệu train và test
data_train = []
data_test = []

# Duyệt qua từng nhãn trong thư mục train
for label in labels_train:
    # Tạo đường dẫn tới thư mục của nhãn hiện tại
    label_path = os.path.join(data_train_path, label)

    # Duyệt qua từng file trong thư mục của nhãn hiện tại
    for filename in os.listdir(label_path):
        # Tạo đường dẫn tới file hiện tại
        file_path = os.path.join(label_path, filename)

        # Đọc hình ảnh từ file
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Thêm dữ liệu và nhãn tương ứng vào danh sách dữ liệu train
        data_train.append((img, label))

# Duyệt qua từng nhãn trong thư mục test
for label in labels_test:
    # Tạo đường dẫn tới thư mục của nhãn hiện tại
    label_path = os.path.join(data_test_path, label)

    # Duyệt qua từng file trong thư mục của nhãn hiện tại
    for filename in os.listdir(label_path):
        # Tạo đường dẫn tới file hiện tại
        file_path = os.path.join(label_path, filename)

        # Đọc hình ảnh từ file
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Thêm dữ liệu và nhãn tương ứng vào danh sách dữ liệu test
        data_test.append((img, label))

# Chuyển đổi danh sách dữ liệu train và test thành numpy array để sử dụng cho việc huấn luyện và đánh giá mô hình
data_train = np.array(data_train, dtype=object)
data_test = np.array(data_test, dtype=object)


# Tiền xử lý dữ liệu train và test
X_train = []
y_train = []
X_test = []
y_test = []

for img, label in data_train:
    # Thực hiện giảm kích thước hình ảnh xuống còn 28x28 pixel
    img_resized = cv2.resize(img, (100, 100))
    # Thực hiện chuyển đổi giá trị pixel từ [0, 255] sang [0, 1]
    img_normalized = img_resized / 255.0
    # Thêm hình ảnh và nhãn tươn ứng vào danh sách X_train và y_train
    X_train.append(img_normalized)
    y_train.append(label)

for img, label in data_test:
    # Thực hiện giảm kích thước hình ảnh xuống còn 28x28 pixel
    img_resized = cv2.resize(img, (100, 100))
    # Thực hiện chuyển đổi giá trị pixel từ [0, 255] sang [0, 1]
    img_normalized = img_resized / 255.0
    # Thêm hình ảnh và nhãn tương ứng vào danh sách X_test và y_test
    X_test.append(img_normalized)
    y_test.append(label)

#Chuyển đổi danh sách X_train, y_train, X_test, y_test thành numpy array để sử dụng cho việc huấn luyện và đánh giá mô hình
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

def extract_hog_features(X):
    hog_features = []
    for img in X:
        # Tính toán đặc trưng HoG cho mỗi hình ảnh
        hog_feature = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        # Thêm đặc trưng HoG của hình ảnh vào danh sách hog_features
        hog_features.append(hog_feature)
    # Chuyển đổi danh sách hog_features thành numpy array
    hog_features = np.array(hog_features)
    return hog_features

# Trích xuất đặc trưng HoG cho tập huấn luyện
X_train_hog = extract_hog_features(X_train)

# Trích xuất đặc trưng HoG cho tập kiểm thử
X_test_hog = extract_hog_features(X_test)

X_train_hog_reshape = X_train_hog.reshape(X_train_hog.shape[0], -1)
X_test_hog_reshape = X_test_hog.reshape(X_test_hog.shape[0], -1)

# Khởi tạo mô hình SVM
clf = svm.SVC(kernel='linear')

# Huấn luyện mô hình trên dữ liệu train
clf.fit(X_train_hog_reshape, y_train)
# # Lưu mô hình vào file svm_model.pkl
joblib.dump(clf, 'svm_model.pkl')

# # Đọc mô hình từ file svm_model.pkl
clf = joblib.load('svm_model.pkl')
# Đánh giá mô hình trên dữ liệu test
accuracy = clf.score(X_test_hog_reshape, y_test)
print("Accuracy: ", accuracy)

