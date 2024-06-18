import numpy as np

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Magic number와 이미지 개수를 읽음
        magic = int.from_bytes(f.read(4), byteorder='big')
        num_images = int.from_bytes(f.read(4), byteorder='big')
        num_rows = int.from_bytes(f.read(4), byteorder='big')
        num_cols = int.from_bytes(f.read(4), byteorder='big')
        
        # 이미지 데이터를 읽고 numpy 배열로 변환
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)
        
        return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Magic number와 레이블 개수를 읽음
        magic = int.from_bytes(f.read(4), byteorder='big')
        num_items = int.from_bytes(f.read(4), byteorder='big')
        
        # 레이블 데이터를 읽고 numpy 배열로 변환
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        return labels


# 데이터 파일 경로
train_images_file = './datasets/mnist/train-images.idx3-ubyte'
train_labels_file = './datasets/mnist/train-labels.idx1-ubyte'
test_images_file = './datasets/mnist/t10k-images.idx3-ubyte'
test_labels_file = './datasets/mnist/t10k-labels.idx1-ubyte'

# MNIST 데이터셋 로드
train_images = load_mnist_images(train_images_file)
train_labels = load_mnist_labels(train_labels_file)
test_images = load_mnist_images(test_images_file)
test_labels = load_mnist_labels(test_labels_file)

# 데이터 확인
print(f'Train images shape: {train_images.shape}')
print(f'Train labels shape: {train_labels.shape}')
print(f'Test images shape: {test_images.shape}')
print(f'Test labels shape: {test_labels.shape}')

print("done")