# 1. Mô tả
Đây là học máy dịch nhãn wikidata từ tiếng anh sang tiếng việt.
Đồ án tốt nghiệp do nhóm 03, sinh viên CTK45 Thực hiện. 
GVHD: ts. Tạ Hoàng Thắng    email: tahoangthang@gmail.com


# 2. Bước thu thập dữ liệu
Thu thập dữ liệu ngẫu nhiên bằng lệnh sau trên terminal:

    python collect_data.py

dữ liệu sẽ được thu thập bằng wikidata api theo id item ngẫu nhiên
# 3. Bước lọc các dữ liệu tên riêng
Chạy lệnh:

    python filtered.py

lọc những dữ liệu có nhãn tiếng anh và tiếng việt trùng lập với nhau.

# 4. chia tập dữ liệu
Tập dữ liệu sẽ được chia ngẫu nhiên theo tỉ lệ 8:1:1, bởi lệnh:

    python split_dataset.py

# 5. Phân tích tập dữ liệu
Lệnh

    python analyze_data.py 

phân tích các tham số
* Độ dài nguồn, độ dài đích (token, spacy)
* số lượng từ ngữ tiếng việt và tiếng anh
* phân loại nhãn theo chủ đề

# 6. các phương pháp huấn luyện

## 6a. Naive Seq2seq

...writing code...

## 6b. mô hình pretrained Transformer

### các mô hình
* Helsinki-NLP/opus-mt-en-vi (https://github.com/Helsinki-NLP/Opus-MT, https://marian-nmt.github.io/)
* VietAI/envit5-translation (https://huggingface.co/VietAI/envit5-translation) 
* mbart (https://huggingface.co/docs/transformers/model_doc/mbart, https://arxiv.org/abs/2001.08210)
* mt5 (https://huggingface.co/google/mt5-base, https://arxiv.org/abs/2010.11934)
* facebook/m2m100_418M (https://huggingface.co/facebook/m2m100_418M, https://arxiv.org/abs/2010.11125)

### tham số mô hình
Danh sách tham số:
* *mode*: train/test/generate
* *epochs*: số lượng epochs
* *batch_size*: kích thước batch
* *test_batch_size*: kích thước test batch
* *max_source_length, max_target_length, min_target_length*: độ dài nguồn và đích
* *model_path*: mô hình huấn luyện (lưu ý chọn đúng checkpoint khi kiểm tra)
* *train_path, val_path, test_path*: Đường dẫn đến tập dữ liệu huấn luyện, tập xác nhận, tập kiểm tra
* *source_prefix*: tiền tố, mô hình t5 yêu cầu tiền tố phù hợp cho các bài toán
* *source_column, target_column*: chỉ định các trường đầu vào và đầu ra trong các tập huấn luyện, xác thực và kiểm tra
* *decode_pred*: Giải mã chuổi dự đoán thành từ tiếng việt hoàn chỉnh (utils.py)
### lệnh chạy huấn luyện/ test mô hình
#### google-t5/t5-small
##### với tiền tố

    python seq2seq.py --mode "train" --model_name "google-t5/t5-small" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "summarize: " --source_column "source" --target_column "target_encoded"

    python seq2seq.py --mode "test" --model_name "google-t5/t5-small" --model_path "google-t5_t5-small\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target" --decode_pred 1

##### Không tiền tố

    python seq2seq.py --mode "train" --model_name "google-t5/t5-small" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "" --source_column "source" --target_column "target_encoded"

    python seq2seq.py --mode "test" --model_name "google-t5/t5-small" --model_path "google-t5_t5-small\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target" --decode_pred 1


#### google-t5/t5-base
##### có tiền tố

    python seq2seq.py --mode "train" --model_name "google-t5/t5-base" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "summarize: " --source_column "source" --target_column "target_encoded"

    python seq2seq.py --mode "test" --model_name "google-t5/t5-base" --model_path "google-t5_t5-base\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target" --decode_pred 1

##### Không tiền tố

    python seq2seq.py --mode "train" --model_name "google-t5/t5-base" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "" --source_column "source" --target_column "target_encoded"

    python seq2seq.py --mode "test" --model_name "google-t5/t5-base" --model_path "google-t5_t5-base\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target" --decode_pred 1

#### facebook/bart-base

    python seq2seq.py --mode "train" --model_name "facebook/bart-base" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "" --source_column "source" --target_column "target_encoded"

    python seq2seq.py --mode "test" --model_name "facebook/bart-base" --model_path "facebook_bart-base\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target" --decode_pred 1

#### Helsinki-NLP/opus-mt-en-vi

    python seq2seq.py --mode "train" --model_name "Helsinki-NLP/opus-mt-en-vi" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "" --source_column "source" --target_column "target"

    python seq2seq.py --mode "test" --model_name "Helsinki-NLP/opus-mt-en-vi" --model_path "opus-mt-en-vi\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target"
 
#### VietAI/envit5-translation

    python seq2seq.py --mode "train" --model_name "VietAI/envit5-translation" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "" --source_column "source" --target_column "target"

    python seq2seq.py --mode "test" --model_name "VietAI/envit5-translation" --model_path "envit5-translation\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target"

#### facebook/m2m100_418M

    python seq2seq.py --mode "train" --model_name "facebook/m2m100_418M" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "" --source_column "source" --target_column "target"

    python seq2seq.py --mode "test" --model_name "facebook/m2m100_418M" --model_path "m2m100_418M\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target"

#### facebook/mbart-large-50 (bad results)

    python seq2seq.py --mode "train" --model_name "facebook/mbart-large-50" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "" --source_column "source" --target_column "target"

    python seq2seq.py --mode "test" --model_name "facebook/mbart-large-50" --model_path "google_mt5-base\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target"

#### google/mt5-base (worst results)

    python seq2seq.py --mode "train" --model_name "google/mt5-base" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "summarize: " --source_column "source" --target_column "target"

    python seq2seq.py --mode "test" --model_name "google/mt5-base" --model_path "google_mt5-base\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "summarize: " --source_column "source" --target_column "target"

* Lưu ý: đổi checkpoint phù hợp với mô hình để test
* Lưu ý 2: nếu có sử dụng máy dùng GPU của nvidia, nên cài đặt phiên bản pytorch phù hợp với phiên bản cuda của bạn: 
- https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html
- https://pytorch.org/get-started/locally/
* Với amd gpu (mò thử chứ ko có card amd để test (～￣▽￣)～ )
- https://pytorch.org/blog/amd-extends-support-for-pt-ml/
- https://www.reddit.com/r/pytorch/comments/1188qd1/amd_gpu_with_pytorch/?rdt=53059

# 7. Phân tích và lọc kết quả 
## 7a. lọc kết quả
lọc ra danh sách câu nguồn và câu dự đoán bằng lệnh

    python filt.py
## 7b. phân tích kết quả

    python analyze.py
* phân tích, so sánh metric
* tỉ lệ chính sát của mô hình
* các lỗi phổ biến
## 7c. giao diện dịch thuật
    python ui.py
* để hiển thị giao diện ứng dụng dịch nhãn từ tiếng anh sang tiếng việt, có thể chọn qua lại các model để kiểm tra và so sánh chất lượng dịch thuật

lưu ý: chọn đúng checkpoint trước khi chạy lệnh

# 8. Liên hệ
- sinh viên thực hiện đồ án
* Nguyễn Việt Linh (vietlinh357@gmail.com, 2115230@dlu.edu.vn)
* Nguyễn Thọ Thành (2115269@dlu.edu.vn)
* Nguyễn Dương Công Bảo (2115288@dlu.edu.vn)

-GVHD
* tahoangthang@gmail.com
* https://www.tahoangthang.com

