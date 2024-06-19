import numpy as np
import os
import pickle

class DATASET_CIFAR10:
    def __init__(self, cifar10_dir='./datasets/cifar10/cifar-10-batches-py'):
        self.cifar10_dir = cifar10_dir
        
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.load_cifar10_data(self.cifar10_dir)

    def load_cifar10_batch(self, filename):
        with open(filename, 'rb') as f:
            # pickle 포맷으로 데이터 로드
            data_dict = pickle.load(f, encoding='bytes')
            images = data_dict[b'data']  # 이미지 데이터
            labels = np.array(data_dict[b'labels'])  # 레이블
            
            # 이미지 데이터 형식 변경: (num_samples, height, width, channels)
            images = images.reshape(-1, 3, 32, 32)  # CIFAR-10 이미지는 32x32 크기의 RGB 이미지임
            # 순서 변경: (num_samples, height, width, channels) -> (num_samples, channels, height, width)
            images = images.transpose(0, 2, 3, 1)
            
            return images, labels

    def load_cifar10_data(self, data_dir):
        train_images = []
        train_labels = []
        for i in range(1, 6):
            filename = os.path.join(data_dir, f'data_batch_{i}')
            images, labels = self.load_cifar10_batch(filename)
            train_images.append(images)
            train_labels.append(labels)
        
        # 훈련 데이터 배열 합치기
        train_images = np.concatenate(train_images, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        
        # 테스트 데이터 읽기
        filename = os.path.join(data_dir, 'test_batch')
        test_images, test_labels = self.load_cifar10_batch(filename)
        
        return train_images, train_labels, test_images, test_labels        

# tmp = DATASET_CIFAR10()
# print(f'Train images shape: {tmp.train_images.shape}')
# print(f'Train labels shape: {tmp.train_labels.shape}')
# print(f'Test  images shape: {tmp.test_images.shape}')
# print(f'Test  labels shape: {tmp.test_labels.shape}')


# def load_cifar10_batch(filename):
#     with open(filename, 'rb') as f:
#         # pickle 포맷으로 데이터 로드
#         data_dict = pickle.load(f, encoding='bytes')
#         images = data_dict[b'data']  # 이미지 데이터
#         labels = data_dict[b'labels']  # 레이블
        
#         # 이미지 데이터 형식 변경: (num_samples, height, width, channels)
#         images = images.reshape(-1, 3, 32, 32)  # CIFAR-10 이미지는 32x32 크기의 RGB 이미지임
#         # 순서 변경: (num_samples, height, width, channels) -> (num_samples, channels, height, width)
#         images = images.transpose(0, 2, 3, 1)
        
#         return images, labels

# def load_cifar10_data(data_dir):
#     train_images = []
#     train_labels = []
#     for i in range(1, 6):
#         filename = os.path.join(data_dir, f'data_batch_{i}')
#         images, labels = load_cifar10_batch(filename)
#         train_images.append(images)
#         train_labels.append(labels)
    
#     # 훈련 데이터 배열 합치기
#     train_images = np.concatenate(train_images, axis=0)
#     train_labels = np.concatenate(train_labels, axis=0)
    
#     # 테스트 데이터 읽기
#     filename = os.path.join(data_dir, 'test_batch')
#     test_images, test_labels = load_cifar10_batch(filename)
    
#     return train_images, train_labels, test_images, test_labels

# # CIFAR-10 데이터셋이 저장된 디렉토리 경로
# cifar10_dir = './datasets/cifar10/cifar-10-batches-py'  # 실제 디렉토리 경로로 변경해야 함

# # CIFAR-10 데이터셋 로드
# train_images, train_labels, test_images, test_labels = load_cifar10_data(cifar10_dir)

# # 데이터 확인
# print(f'Train images shape: {train_images.shape}')
# print(f'Train labels shape: {len(train_labels)}')
# print(f'Test images shape: {test_images.shape}')
# print(f'Test labels shape: {len(test_labels)}')