import json
import matplotlib.pyplot as plt
import numpy as np

# Danh sách file kết quả của các mô hình
files = {
    "T5": "dataset/test_pred_t5-base.json",
    "BART": "dataset/test_pred_bart-base.json",
    "OPUS-MT": "dataset/test_pred_opus-mt-en-vi.json"
}

# Hàm tính Accuracy cho từng mô hình
def compute_accuracy(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if "source" in line]

    correct = sum(1 for sample in test_data if sample["prediction"] == sample["target"])
    total = len(test_data)
    accuracy = (correct / total) * 100 if total > 0 else 0

    return accuracy, correct, total

# Tính Accuracy cho từng mô hình
accuracy_scores = {}
correct_counts = {}
total_counts = {}

for model, file in files.items():
    acc, correct, total = compute_accuracy(file)
    accuracy_scores[model] = acc
    correct_counts[model] = correct
    total_counts[model] = total
    print(f"📌 {model} - Accuracy: {acc:.2f}% ({correct}/{total} đúng)")

# Vẽ biểu đồ so sánh độ chính xác
models = list(accuracy_scores.keys())
accuracies = list(accuracy_scores.values())

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'red', 'green'])
plt.ylabel("Độ chính xác (%)")
plt.xlabel("Mô hình")
plt.title("🎯 So sánh độ chính xác của mô hình T5, BART và OPUS-MT")
plt.ylim(0, 100)  # Giới hạn từ 0 - 100%
plt.show()


import json
import nltk
import evaluate
from nltk.translate.bleu_score import corpus_bleu

# Load các thư viện đánh giá
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

import json
import matplotlib.pyplot as plt
import numpy as np

# Danh sách file kết quả của các mô hình
files = {
    "T5": "dataset/test_pred_t5-base.json",
    "BART": "dataset/test_pred_bart-base.json",
    "OPUS-MT": "dataset/test_pred_opus-mt-en-vi.json"
}

# Hàm lấy điểm số từ file JSON
def extract_scores(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in reversed(lines):  # Duyệt từ cuối lên để tìm "test_results"
        data = json.loads(line.strip())
        if "test_results" in data:
            return data["test_results"]

    return None  # Trả về None nếu không tìm thấy

# Lấy điểm số từ từng mô hình
scores = {model: extract_scores(file) for model, file in files.items()}

# Kiểm tra xem có mô hình nào bị lỗi không
for model, result in scores.items():
    if result is None:
        print(f"⚠ Không tìm thấy `test_results` trong file {files[model]}!")

# Hiển thị điểm số của từng mô hình
print(json.dumps(scores, indent=4))
# Chuẩn bị dữ liệu
metrics = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"]
t5_scores = [scores["T5"]["bleu"], scores["T5"]["rouge1"], scores["T5"]["rouge2"], scores["T5"]["rougeL"], scores["T5"]["meteor"]]
bart_scores = [scores["BART"]["bleu"], scores["BART"]["rouge1"], scores["BART"]["rouge2"], scores["BART"]["rougeL"], scores["BART"]["meteor"]]
opus_scores = [scores["OPUS-MT"]["bleu"], scores["OPUS-MT"]["rouge1"], scores["OPUS-MT"]["rouge2"], scores["OPUS-MT"]["rougeL"], scores["OPUS-MT"]["meteor"]]

# Thiết lập vị trí trên trục x
x = np.arange(len(metrics))
width = 0.2  # Độ rộng của cột

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.bar(x - width, t5_scores, width, label="T5", color="blue")
plt.bar(x, bart_scores, width, label="BART", color="red")
plt.bar(x + width, opus_scores, width, label="OPUS-MT", color="green")

# Thêm nhãn
plt.xlabel("Metric")
plt.ylabel("Điểm số")
plt.title("📊 So sánh BLEU, ROUGE, METEOR giữa T5, BART, OPUS-MT")
plt.xticks(x, metrics)
plt.legend()
plt.ylim(0, 1)  # Giới hạn từ 0 - 1
plt.show()

import json
import matplotlib.pyplot as plt
from collections import Counter



# Hàm lấy danh sách lỗi từ file JSON
def extract_errors(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(line.strip()) for line in f]

    errors = []
    for sample in test_data:
        # Kiểm tra nếu "prediction" hoặc "target" không tồn tại, bỏ qua dòng này
        if "prediction" not in sample or "target" not in sample:
            continue

        if sample["prediction"] != sample["target"]:  # Nếu dự đoán sai
            error_text = f"❌ {sample['prediction']} (thay vì {sample['target']})"
            errors.append(error_text)

    return errors


# Tạo danh sách lỗi từ từng mô hình
errors_all = {model: extract_errors(file) for model, file in files.items()}

# Gộp lỗi từ tất cả mô hình
all_errors = errors_all["T5"] + errors_all["BART"] + errors_all["OPUS-MT"]

# Đếm tần suất các lỗi phổ biến
error_count = Counter(all_errors).most_common(10)  # Lấy 10 lỗi phổ biến nhất

# In ra top lỗi phổ biến
print("🔍 Top lỗi phổ biến nhất:")
for error, count in error_count:
    print(f"{error}: {count} lần")


# Vẽ biểu đồ tần suất lỗi
plt.figure(figsize=(10, 5))
plt.barh([x[0] for x in error_count], [x[1] for x in error_count], color='orange')
plt.xlabel("Số lần lỗi")
plt.ylabel("Lỗi phổ biến")
plt.title("🔍 Top 10 lỗi phổ biến trong kết quả dịch")
plt.gca().invert_yaxis()  # Đảo ngược thứ tự để lỗi xuất hiện nhiều nhất nằm trên cùng
plt.show()
