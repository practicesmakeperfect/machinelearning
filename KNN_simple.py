import matplotlib
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 设置后端为 TkAgg

image_file = r'D:\edge\archive\train-images.idx3-ubyte'
label_file = r'D:\edge\archive\train-labels.idx1-ubyte'

test_file_x = r'D:\edge\archive\t10k-images.idx3-ubyte'
test_file_y = r'D:\edge\archive\t10k-labels.idx1-ubyte'

m_x = idx2numpy.convert_from_file(image_file)
m_y = idx2numpy.convert_from_file(label_file)
m_x = m_x[:100]
m_y = m_y[:100]
test_x = idx2numpy.convert_from_file(test_file_x)

test_y = idx2numpy.convert_from_file(test_file_y)
def distance(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))

class KNN:
    def __init__(self, k, label_num):
        self.k = k
        self.label_num = label_num
        self.x_train = []
        self.y_train = []

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def get_nearest_neighbors(self, x_test):
        distances = []
        for i in range(len(self.x_train)):
            dist = distance(self.x_train[i], x_test)
            distances.append((dist, self.y_train[i]))
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]

    def predict(self, x_test):
        knn = self.get_nearest_neighbors(x_test)
        labels = [0]*self.label_num
        for i in range(len(knn)):
            labels[knn[i][1]] += 1
        return np.argmax(labels)

for k in range(1, 10):
    p = KNN(k, 10)
    p.fit(m_x, m_y)
    correct = 0
    for i in range(len(test_x)):
        pre_label = p.predict(test_x[i])
        if pre_label == test_y[i]:
            correct += 1
    accuracy = correct / len(test_x)
    print(f"Accuracy for K={k}: {accuracy*100:.2f}%")
