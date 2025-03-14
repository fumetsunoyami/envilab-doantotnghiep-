import json

# Định nghĩa đường dẫn file (thay đổi nếu cần)
input_file = "dataset/collected_data.json"      # File gốc chứa dữ liệu
output_file = "dataset/filt_data.json"  # File sau khi lọc


filtered_data = []
with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        try:
            entry = json.loads(line.strip())  # Đọc từng dòng JSON
            if entry["target_encoded"] != entry["source"]:  # Lọc dữ liệu
                filtered_data.append(entry)
        except json.JSONDecodeError as e:
            print(f"Lỗi JSON ở dòng {i}: {e}")  # In lỗi nhưng không dừng chương trình

# Ghi dữ liệu đã lọc vào file mới
with open(output_file, "w", encoding="utf-8") as f:
    for entry in filtered_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Đã lọc xong! Số lượng dòng còn lại: {len(filtered_data)}")

