import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical

# Đường dẫn tới thư mục chứa dữ liệu train và test
data_train_path = 'DATA/train'
data_test_path = 'DATA/test'

# Khởi tạo danh sách tên các nhãn (tên thư mục) trong thư mục train
labels_train = os.listdir(data_train_path)

# Khởi tạo danh sách tên các nhãn (tên thư mục) trong thư mục test
labels_test = os.listdir(data_test_path)

# Create empty arrays for the images and labels
X_train = []
y_train = []
X_test = []
y_test = []

# Loop over the images in the train set
for label in labels_train:
    label_path = os.path.join(data_train_path, label)
    for filename in os.listdir(label_path):
        file_path = os.path.join(label_path, filename)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        X_train.append(img)
        y_train.append(label)

# Loop over the images in the test set
for label in labels_test:
    label_path = os.path.join(data_test_path, label)
    for filename in os.listdir(label_path):
        file_path = os.path.join(label_path, filename)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        X_test.append(img)
        y_test.append(label)

# Convert the lists to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# # Chuyển đổi định dạng của dữ liệu và nhãn
# X_train = X_train.astype('float32') / 255.
# X_test = X_test.astype('float32') / 255.
# y_train = to_categorical(y_train, num_classes=len(labels_train))
# y_test = to_categorical(y_test, num_classes=len(labels_test))

#Thay đổi kích thước của dữ liệu train và test thành (width, height, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Khởi tạo mô hình CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(labels_train), activation='softmax'))

#Biên dịch mô hình CNN với hàm mất mát categorical_crossentropy và tối ưu hóa Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Đào tạo mô hình CNN trên dữ liệu train
model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1, validation_data=(X_test, y_test))

# Đánh giá mô hình CNN trên dữ liệu test
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])