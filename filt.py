import json

# Danh sách file kết quả của các mô hình
files = {
    "T5": "dataset/test_pred_t5-base.json",
    "BART": "dataset/test_pred_bart-base.json",
    "OPUS-MT": "dataset/test_pred_opus-mt-en-vi.json"
}

# Hàm lọc dữ liệu
def filter_data(file_path):
    filtered_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line.strip())
            if "source" in sample and "prediction" in sample:
                filtered_data.append({"source": sample["source"], "prediction": sample["prediction"]})
    return filtered_data

# Lọc dữ liệu từ từng mô hình
filtered_results = {model: filter_data(file) for model, file in files.items()}

# Kiểm tra kết quả (in 3 dòng đầu tiên của mỗi mô hình)
for model, data in filtered_results.items():
    print(f"\n📌 {model} - Dữ liệu mẫu:")
    print(json.dumps(data[:3], indent=4, ensure_ascii=False))
# Lưu dữ liệu đã lọc vào file JSON mới
for model, data in filtered_results.items():
    output_file = f"filtered_{model}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ Dữ liệu đã lọc của {model} được lưu vào {output_file}")
