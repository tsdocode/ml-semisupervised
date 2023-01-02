# Tiểu luận chuyên ngành - Kỹ thuật dữ liệu - 2022
## Đề tài: Tìm hiểu về semi-supervised learning và ứng dụng

> 
    Nguyễn Thanh Sang   - 19133048
    Lê Thị Nhung        - 19133043

## 1. Các bước chạy lại thí nghiệm
### 1.1 Cài đặt các thư viện
```bash
pip install -r requirements.txt
```

### 1.2. Khởi chạy các thí nghiệm trong notebook Experiment.ipynb

### 1.3. Theo dỏi mô tả các mô hình
```bash
mlflow ui
```
Truy cập giao diện theo dõi của MLflows: http://localhost:5000


## 2. Thông tin các thí nghiệm
### 2.1 Các tập dữ liệu 
- MNIST
- MUSHROOM
- SPAM TEXT
### 2.2 Setup dữ liệu cho các thí nghiệm
- Dữ liệu được phân chia train/test với tỉ lệ **80%/20%**
- tỉ lệ các điểm dữ liệu được gán nhãn lần lượt là: **5%->10%->20%->50%**

### 2.2 Thuật toán
- Self-training
- Contraining & Multiview training
- Gaussian Mixture Model
- Graph-based method using harmonic function
- Transductive SVM


### 3. Cấu trúc code

```
|---dataset
    |---file
        |---mushroom.csv    # Mushroom dataset
        |---spam_text.csv   # Spam text dataset
    |---ssl_dataset.py      # Main class of all SSL dataset
    |---mnist.py            # MNIST dataset handler
    |---mushroom.py         # MUSHROOM dataset handler
    |---spam_text.py        # SPAM TEXT dataset handler
|---experiment
    |---ssl_experiment.py                   # Main class of all SSL experiments
    |---self_training_experiment.py         # Self training experiment
    |---multiview_training_experiment.py    # Co-training & Multiview experiment
    |---gmm_experiment.py                   # Gaussian Mixture Model experiment
    |---harmonic_experiment.py              # Graph-based method experiment
    |---s3vm_experiment.py                  # Semi-supervised Support-vector machine experiment
|---model
    |---graph_harmonic.py                   # Graph-based usign harmonic function model
    |---s3vm.py                             # Semi-supervised SVM model
|---ref                                     # References resource

```