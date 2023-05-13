# Bài tập cuối kỳ
Đề bài: Tìm một dữ liệu và áp dụng các phương pháp học của mình vào

# Dataset
- link tải dữ liệu: https://www.kaggle.com/datasets/vikrishnan/iris-dataset
- Dữ liệu phân lớp
- Dự vào mô hình phân tách tuyến tính Fisher
- Là tập phân lớp tiêu chuẩn cho các phương phân lớp dựa vào thống kê.

## Nội dung dữ liệu
- Dữ liệu hoa diên vĩ là dữ liệu đa biến (multivariable data set)
- Dữ liệu gồm 50 mẫu cho mỗi 3 lớp:
    + Iris setosa
    + Iris virginica
    + Iris versicolor

- 4 thuộc tính bao gồm:
    1. độ dài đài hoa
    2. Chiều rộng đài hoa
    3. Chiều dài cánh hoa
    4. Chiều rộng cánh hoa

# Tài liệu:
- Notion: https://outrageous-jonquil-e0d.notion.site/B-i-t-p-cu-i-k-cd50e2855167473f8ef534a908f25f66
- Sách trí tuệ nhân tạo phần 1: https://drive.google.com/file/d/1O1lHRMmW3w5mUTSPZLWQAM-2rnsrBf_s/view
- Sách trí tuệ nhân tạo phần 2: https://drive.google.com/file/d/1135j8osukbcvtT3EL-ZXyshBJfTpSjb9/view





# Yêu cầu của thầy Thúc
Khi report lab 4, phân tích dữ liệu, cần làm nổi bật
- bản chất dữ liệu
- tiếp cận và pp
- tại sao sử dụng pp đó
- bình luận về kết quả và mở rộng nếu có

# Khám phá dữ liệu




# Các phương pháp

Nhóm liệt kê và so sánh một vài phương pháp khớp dữ liệu (data fitting):
- Chương 2: khớp dữ liệu tuyến tính (linear fitting).
- Chương 3: mô hình hồi quy Logistic.
- Chương 4: khớp dữ liệu với mô hình Perceptron.

Trong thực nghiệm, kỹ thuật học đạo hàm ngược (gradient descent) được dùng chung cho cả 2 mô hình hồi quy Logistic và mô hình Perceptron. Trong tập dữ liệu rất nhỏ là Iris này, một kỹ thuật học đơn giản hơn là khớp dữ liệu tuyến tính vẫn cho kết quả tương đương với mô hình hồi quy Logistic và mô hình Perceptron. Trong tương lai, chúng tôi dự tính sẽ so sánh các phương pháp trong 1 tập dữ liệu lớn hơn, mà ở đó kỳ vọng kỹ thuật học Đạo hàm ngược sẽ cho kết quả vượt trội.