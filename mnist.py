import numpy as np

class DATASET_MNIST:
    def __init__(self, mnist_dir="./datasets/mnist/"):
        self.train_images_file = mnist_dir + 'train-images.idx3-ubyte'
        self.train_labels_file = mnist_dir + 'train-labels.idx1-ubyte'
        self.test_images_file  = mnist_dir + 't10k-images.idx3-ubyte'
        self.test_labels_file  = mnist_dir + 't10k-labels.idx1-ubyte'
        
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        
        self.load_mnist()
    
    def load_mnist(self):
        self.train_images = self.load_mnist_images(self.train_images_file)
        self.train_labels = self.load_mnist_labels(self.train_labels_file)
        self.test_images  = self.load_mnist_images (self.test_images_file)
        self.test_labels  = self.load_mnist_labels (self.test_labels_file)    
    
    def load_mnist_images(self, filename):
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

    def load_mnist_labels(self, filename):
        with open(filename, 'rb') as f:
            # Magic number와 레이블 개수를 읽음
            magic = int.from_bytes(f.read(4), byteorder='big')
            num_items = int.from_bytes(f.read(4), byteorder='big')
            
            # 레이블 데이터를 읽고 numpy 배열로 변환
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            
            return labels
        
tmp = DATASET_MNIST()
print(f'Train images shape: {tmp.train_images.shape}')
print(f'Train labels shape: {tmp.train_labels.shape}')
print(f'Test images shape:  {tmp.test_images.shape}')
print(f'Test labels shape:  {tmp.test_labels.shape}')
        
# def load_mnist_images(filename):
#     with open(filename, 'rb') as f:
#         # Magic number와 이미지 개수를 읽음
#         magic = int.from_bytes(f.read(4), byteorder='big')
#         num_images = int.from_bytes(f.read(4), byteorder='big')
#         num_rows = int.from_bytes(f.read(4), byteorder='big')
#         num_cols = int.from_bytes(f.read(4), byteorder='big')
        
#         # 이미지 데이터를 읽고 numpy 배열로 변환
#         images = np.frombuffer(f.read(), dtype=np.uint8)
#         images = images.reshape(num_images, num_rows, num_cols)
        
#         return images

# def load_mnist_labels(filename):
#     with open(filename, 'rb') as f:
#         # Magic number와 레이블 개수를 읽음
#         magic = int.from_bytes(f.read(4), byteorder='big')
#         num_items = int.from_bytes(f.read(4), byteorder='big')
        
#         # 레이블 데이터를 읽고 numpy 배열로 변환
#         labels = np.frombuffer(f.read(), dtype=np.uint8)
        
#         return labels


# # 데이터 파일 경로
# train_images_file = './datasets/mnist/train-images.idx3-ubyte'
# train_labels_file = './datasets/mnist/train-labels.idx1-ubyte'
# test_images_file = './datasets/mnist/t10k-images.idx3-ubyte'
# test_labels_file = './datasets/mnist/t10k-labels.idx1-ubyte'

# # MNIST 데이터셋 로드
# train_images = load_mnist_images(train_images_file)
# train_labels = load_mnist_labels(train_labels_file)
# test_images = load_mnist_images(test_images_file)
# test_labels = load_mnist_labels(test_labels_file)

# # 데이터 확인
# print(f'Train images shape: {train_images.shape}')
# print(f'Train labels shape: {train_labels.shape}')
# print(f'Test images shape: {test_images.shape}')
# print(f'Test labels shape: {test_labels.shape}')

# print("done")