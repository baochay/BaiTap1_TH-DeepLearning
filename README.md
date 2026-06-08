# DoAn1_TH-DeepLearning# Handwritten Digit Recognition using CNN (MNIST & Real-world Data)

Ứng dụng nhận diện chữ số viết tay từ `0` đến `9` sử dụng mạng thần kinh tích chập (CNN) với thư viện **PyTorch**. 
Dự án không chỉ huấn luyện trên bộ dữ liệu chuẩn **MNIST** mà còn hỗ trợ tiền xử lý và huấn luyện nâng cao (Fine-tune) với dữ liệu chữ số chụp thực tế.


## 📊 Kết quả huấn luyện (Training Metrics)
Mô hình đạt hiệu suất cực cao sau 20 chu kỳ huấn luyện (Epochs) trên tập MNIST:
- **Độ chính xác tập Huấn luyện (Train Accuracy):** 99.78%
- **Độ chính xác tập Kiểm thử (Test Accuracy):** **99.22%**
- **Hàm mất mát tập Kiểm thử (Val Loss):** 0.0458 (Mô hình hội tụ tốt, không bị Overfitting).

---

## 📁 Cấu trúc thư mục dự án
```text
MNIST_Code/
├── data/                          # Thư mục chứa bộ dữ liệu MNIST gốc, ảnh thực tế và ảnh thực tế đã qua xử lý
├── models/                        # Thư mục lưu trữ trọng số mô hình
│   ├── best_model.pth             # Model có độ chính xác cao nhất từ MNIST
│   └── training_metrics.png       # Đồ thị biểu diễn Loss và Accuracy
├── src/                           # Thư mục chứa mã nguồn chính
│   ├── data_collected.py          # Xử lý thu thập ảnh thực tế 
│   ├── data_loader.py             # Xử lý và phân chia dữ liệu (Train/Val/Test)
│   ├── model.py                   # Định nghĩa cấu trúc mạng CNN
│   ├── train.py                   # Huấn luyện mô hình gốc trên MNIST
│   └── train_finetune.py          # Huấn luyện nâng cao (Fine-tune) với ảnh thực tế\
├── app.py                         # File chạy ứng dụng giao diện
└── preprocess_real_images.py      # Tiền xử lý ảnh chụp thực tế 
