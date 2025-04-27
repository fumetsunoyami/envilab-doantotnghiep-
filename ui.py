import tkinter as tk
from tkinter import scrolledtext, ttk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# Danh sách ký tự tiếng Việt mã hóa
VICODE = ["À", "Á", "Â", "Ã", "È", "É", "Ê", "Ì", "Í", "Ò","Ó", "Ô", "Õ", 
        "Ù", "Ú", "Ý", "à", "á", "â", "ã", "è", "é", "ê", "ì", "í", "ò", "ó", 
        "ô", "õ", "ù", "ú", "ý", "Ă", "ă", "Đ", "đ", "Ĩ", "ĩ", "Ũ", "ũ",
			"Ơ", "ơ", "Ư", "ư", "Ạ", "ạ", "Ả", "ả", "Ấ", "ấ",
			"Ầ", "ầ", "Ẩ", "ẩ", "Ẫ", "ẫ", "Ậ", "ậ", "Ắ", "ắ",
			"Ằ", "ằ", "Ẳ", "ẳ", "Ẵ", "ẵ", "Ặ", "ặ", "Ẹ", "ẹ",
			"Ẻ", "ẻ", "Ẽ", "ẽ", "Ế", "ế", "Ề", "ề", "Ể", "ể",
			"Ễ", "ễ", "Ệ", "ệ", "Ỉ", "ỉ", "Ị", "ị", "Ọ", "ọ",
			"Ỏ", "ỏ", "Ố", "ố", "Ồ", "ồ", "Ổ", "ổ", "Ỗ", "ỗ",
			"Ộ", "ộ", "Ớ", "ớ", "Ờ", "ờ", "Ở", "ở", "Ỡ", "ỡ",
			"Ợ", "ợ", "Ụ", "ụ", "Ủ", "ủ", "Ứ", "ứ", "Ừ", "ừ",
			"Ử", "ử", "Ữ", "ữ", "Ự", "ự", "Ỳ", "ỳ", "Ỵ", "ỵ",
			"Ỷ", "ỷ", "Ỹ", "ỹ"]

def decode_vi(txt):
    """ Giải mã ký tự tiếng Việt từ dạng @xx về ký tự gốc """
    re_lst = re.findall('@[0-9]+', txt)
    for x in re_lst:
        txt = txt.replace(x, VICODE[int(x[1:])])
    return txt



# Danh sách mô hình và checkpoint tương ứng
MODEL_OPTIONS = {
    
    "OPUS-MT": ("Helsinki-NLP/opus-mt-en-vi", "opus-mt-en-vi/checkpoint-450"),
    "BART": ("facebook/bart-base", "bart-base/checkpoint-2550"),
    "T5": ("google-t5/t5-base", "t5-base/checkpoint-2475"),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

current_tokenizer = None
current_model = None
current_direction = "en-vi"

# Tải mô hình ban đầu
model_name, model_path = MODEL_OPTIONS["OPUS-MT"]
current_tokenizer = AutoTokenizer.from_pretrained(model_path)
current_model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
current_model.eval()

def load_model(selection):
    global current_model, current_tokenizer
    model_name, model_path = MODEL_OPTIONS[selection]
    current_tokenizer = AutoTokenizer.from_pretrained(model_path)
    current_model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    current_model.eval()

def translate_text():
    input_text = input_box.get("1.0", tk.END).strip()
    if not input_text:
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, "Vui lòng nhập văn bản để dịch!")
        return


    inputs = current_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)

    with torch.no_grad():
        outputs = current_model.generate(**inputs, max_length=128, num_beams=4)

    translated_text = current_tokenizer.decode(outputs[0], skip_special_tokens=True)
    translated_text = decode_vi(translated_text)

    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, translated_text)



# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Ứng dụng dịch nhãn (Wikidata Translator)")
root.geometry("700x500")

# Mô hình lựa chọn
tk.Label(root, text="Chọn mô hình:", font=("Arial", 11)).pack()
model_selector = ttk.Combobox(root, values=list(MODEL_OPTIONS.keys()), state="readonly")
model_selector.set("OPUS-MT")
model_selector.pack(pady=5)
model_selector.bind("<<ComboboxSelected>>", lambda e: load_model(model_selector.get()))


# Ô nhập văn bản
tk.Label(root, text="Nhập văn bản cần dịch:", font=("Arial", 12)).pack()
input_box = scrolledtext.ScrolledText(root, height=5, wrap=tk.WORD, font=("Arial", 12))
input_box.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

# Nút dịch
tk.Button(root, text="Dịch", font=("Arial", 12), command=translate_text).pack(pady=10)

# Kết quả
tk.Label(root, text="Kết quả dịch:", font=("Arial", 12)).pack()
output_box = scrolledtext.ScrolledText(root, height=6, wrap=tk.WORD, font=("Arial", 12), fg="blue")
output_box.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

root.mainloop()

'''from transformers import AutoTokenizer

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
root.mainloop()'''