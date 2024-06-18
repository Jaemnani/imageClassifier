import numpy as np
import os
import pickle

def load_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        # pickle 포맷으로 데이터 로드
        data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']  # 이미지 데이터
        labels = data_dict[b'labels']  # 레이블
        
        # 이미지 데이터 형식 변경: (num_samples, height, width, channels)
        images = images.reshape(-1, 3, 32, 32)  # CIFAR-10 이미지는 32x32 크기의 RGB 이미지임
        # 순서 변경: (num_samples, height, width, channels) -> (num_samples, channels, height, width)
        images = images.transpose(0, 2, 3, 1)
        
        return images, labels

def load_cifar10_data(data_dir):
    images_train = []
    labels_train = []
    for i in range(1, 6):
        filename = os.path.join(data_dir, f'data_batch_{i}')
        images, labels = load_cifar10_batch(filename)
        images_train.append(images)
        labels_train.append(labels)
    
    # 훈련 데이터 배열 합치기
    images_train = np.concatenate(images_train, axis=0)
    labels_train = np.concatenate(labels_train, axis=0)
    
    # 테스트 데이터 읽기
    filename = os.path.join(data_dir, 'test_batch')
    images_test, labels_test = load_cifar10_batch(filename)
    
    return images_train, labels_train, images_test, labels_test

# CIFAR-10 데이터셋이 저장된 디렉토리 경로
cifar10_dir = './datasets/cifar10/cifar-10-batches-py'  # 실제 디렉토리 경로로 변경해야 함

# CIFAR-10 데이터셋 로드
images_train, labels_train, images_test, labels_test = load_cifar10_data(cifar10_dir)

# 데이터 확인
print(f'Train images shape: {images_train.shape}')
print(f'Train labels shape: {len(labels_train)}')
print(f'Test images shape: {images_test.shape}')
print(f'Test labels shape: {len(labels_test)}')