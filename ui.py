from transformers import AutoTokenizer

part = "opus-mt-en-vi/checkpoint-500"

model_name = "Helsinki-NLP/opus-mt-en-vi" #"facebook/bart-base"  #"google-t5/t5-base" "Helsinki-NLP/opus-mt-en-vi"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Lưu lại vào checkpoint của bạn
tokenizer.save_pretrained(part)

import tkinter as tk
from tkinter import scrolledtext
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Tải checkpoint của mô hình đã huấn luyện
MODEL_PATH = part # Đổi đường dẫn đến checkpoint của bạn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load mô hình và tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

def translate_text():
    """ Dịch văn bản từ tiếng Anh sang tiếng Việt """
    input_text = input_box.get("1.0", tk.END).strip()  # Lấy nội dung nhập vào
    if not input_text:
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, "Vui lòng nhập văn bản để dịch!")
        return
    
    # Token hóa đầu vào
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)

    # Dịch bằng mô hình
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4)

    # Giải mã kết quả
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Hiển thị kết quả trong ô output
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, translated_text)

# Tạo cửa sổ Tkinter
root = tk.Tk()
root.title("English to Vietnamese Translator")
root.geometry("600x400")

# Tiêu đề
tk.Label(root, text="Nhập văn bản tiếng Anh:", font=("Arial", 12)).pack(pady=5)

# Ô nhập văn bản
input_box = scrolledtext.ScrolledText(root, height=5, wrap=tk.WORD, font=("Arial", 12))
input_box.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

# Nút Dịch
translate_button = tk.Button(root, text="Dịch", font=("Arial", 12), command=translate_text)
translate_button.pack(pady=10)

# Ô hiển thị kết quả
tk.Label(root, text="Kết quả dịch:", font=("Arial", 12)).pack(pady=5)
output_box = scrolledtext.ScrolledText(root, height=5, wrap=tk.WORD, font=("Arial", 12), fg="blue")
output_box.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

# Chạy giao diện
root.mainloop()
