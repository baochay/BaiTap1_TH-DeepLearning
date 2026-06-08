from PIL import Image, ImageEnhance
import numpy as np
import os


def crop_auto(img):
    # Cắt viền trắng tự động
    img_np = np.array(img)
    mask = img_np < 255  # vùng không phải trắng hoàn toàn
    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1  # slice end is exclusive
        img_cropped = img.crop((x0, y0, x1, y1))
        return img_cropped
    else:
        return img


def enhance_image(img, contrast=1.2, brightness=1.1):
    # Tăng độ tương phản
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)

    # Tăng độ sáng
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness)

    return img


def preprocess_images(input_dir, output_dir, size=(28, 28), enhance=True):
    """Xử lý toàn bộ ảnh trong thư mục đầu vào và lưu vào thư mục đầu ra (ĐÃ BỎ SƠN LƯU)"""
    if not os.path.exists(input_dir):
        print(f"Thư mục đầu vào không tồn tại: {input_dir}")
        return

    # Tính tổng số ảnh
    total_images = 0
    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        if os.path.isdir(label_dir):
            total_images += len(
                [f for f in os.listdir(label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    if total_images == 0:
        print("Không tìm thấy ảnh nào để xử lý.")
        return

    print(f"Bắt đầu tiền xử lý dữ liệu thực tế từ: {input_dir}")

    processed_images = 0
    error_images = []

    # Duyệt qua từng thư mục nhãn (0 -> 9)
    for label in sorted(os.listdir(input_dir)):
        label_dir = os.path.join(input_dir, label)
        out_label_dir = os.path.join(output_dir, label)

        if not os.path.isdir(label_dir):
            continue

        os.makedirs(out_label_dir, exist_ok=True)

        # Xử lý từng ảnh
        for img_name in os.listdir(label_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): \
                    continue

            img_path = os.path.join(label_dir, img_name)
            try:
                # Đọc và xử lý ảnh
                img = Image.open(img_path).convert('L')
                img = crop_auto(img)
                if enhance:
                    img = enhance_image(img)
                img = img.resize(size)

                # Lưu ảnh đã xử lý
                img.save(os.path.join(out_label_dir, img_name))
                processed_images += 1

            except Exception as e:
                error_images.append((img_path, str(e)))

    # Báo cáo kết quả
    print("\nKết quả xử lý:")
    print(f"- Tổng số ảnh: {total_images}")
    print(f"- Đã xử lý thành công: {processed_images}")
    print(f"- Số ảnh lỗi: {len(error_images)}")

    if error_images:
        print("\nDanh sách ảnh lỗi:")
        for path, err in error_images[:5]:
            print(f"- {os.path.basename(path)}: {err}")
        if len(error_images) > 5:
            print(f"... và {len(error_images) - 5} ảnh lỗi khác.")