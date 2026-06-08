import os
import sys
import torch
from torchvision.utils import save_image

# Định tuyến hệ thống để nhận diện được thư mục src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Gọi hàm load_data từ data_loader của bạn
from src.data_loader import load_data


def export_test_images(num_images=30):
    """
    Trích xuất một số lượng ảnh ngẫu nhiên từ tập MNIST Test và lưu thành file .png
    """
    output_dir = os.path.join(current_dir, 'test_images')
    os.makedirs(output_dir, exist_ok=True)

    print("⏳ Đang nạp dữ liệu MNIST để trích xuất ảnh test...")
    # Gọi hàm load_data (bộ data_loader của bạn sẽ tự tải nếu chưa có hoặc dùng data có sẵn)
    _, _, test_loader, _ = load_data(data_dir='data', batch_size=1)

    print(f"📸 Tiến hành lưu {num_images} ảnh test vào thư mục '{output_dir}'...")

    count = 0
    for images, labels in test_loader:
        if count >= num_images:
            break

        label = labels[0].item()
        # Đặt tên file chứa cả số thứ tự và nhãn đúng (label) để bạn dễ đối chiếu khi test app
        file_name = f"image_{count}_label_{label}.png"
        file_path = os.path.join(output_dir, file_name)

        # Đưa ảnh về khoảng giá trị [0, 1] để lưu ảnh chuẩn bằng torchvision
        img_to_save = images[0] * 0.5 + 0.5  # Phục hồi từ Normalize(-0.5/0.5) nếu có

        save_image(img_to_save, file_path)
        print(f" -> Đã lưu: {file_name}")
        count += 1

    print(f"\n[THÀNH CÔNG] Đã tạo xong {num_images} ảnh test sạch tại thư mục: test_images/")


if __name__ == '__main__':
    export_test_images(num_images=30)  # Bạn có thể đổi số này nếu muốn tải nhiều ảnh hơn