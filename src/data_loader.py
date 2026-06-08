import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_data(data_dir='data', batch_size=32):
    #(Tiền xử lý)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Chuyển ảnh thành Tensor và tự động chuẩn hóa về khoảng [0.0, 1.0]
        transforms.Normalize((0.5,), (0.5,))  # Chuẩn hóa dữ liệu về khoảng [-1.0, 1.0]
    ])

    # Tải tập dữ liệu gốc Train và Test trực tiếp từ torchvision
    full_train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # MNIST train có 60,000 ảnh -> Chia 80% Train (48,000 ảnh) và 20% Validation (12,000 ảnh)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Sử dụng hàm random_split của PyTorch để chia ngẫu nhiên với seed cố định
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    num_classes = 10

    # Tạo các DataLoader để băm nhỏ dữ liệu thành các Batch khi huấn luyện
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes


if __name__ == "__main__":
    # Chạy thử nghiệm để kiểm tra luồng dữ liệu
    train_loader, val_loader, test_loader, num_classes = load_data()
    print("Dữ liệu MNIST đã được tải và xử lý thành công!")
    print(f"Số lượng batches tập Train: {len(train_loader)}")
    print(f"Số lượng batches tập Validation: {len(val_loader)}")
    print(f"Số lượng batches tập Test: {len(test_loader)}")
    print(f"Tổng số lớp phân loại: {num_classes}")