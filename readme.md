# 0. Papers
* Paper: https://arxiv.org/pdf/2408.06124
  
# 1. Introduction
This repo is to translate Wikipedia categories from English to Vietnamese.

# 2. Collect data

## 2a. Collect categories randomly
python collect_data.py

## 2b. Scan all data
...writing...

# 3. Split dataset
Split the dataset randomly by a ratio 8:1:1, by this command:

python split_dataset.py

# 4. Dataset analyses
...writing...
* Length of source, target (tokens, spaCy)
* The number of vocabularies
* Classify categories by topics

# 5. Methods

## 5a. Naive Seq2seq

...writing code...

## 5b. Transformer pretrained models

### Models
* Helsinki-NLP/opus-mt-en-vi (https://github.com/Helsinki-NLP/Opus-MT, https://marian-nmt.github.io/)
* VietAI/envit5-translation (https://huggingface.co/VietAI/envit5-translation)
* mbart (https://huggingface.co/docs/transformers/model_doc/mbart, https://arxiv.org/abs/2001.08210)
* mt5 (https://huggingface.co/google/mt5-base, https://arxiv.org/abs/2010.11934)
* facebook/m2m100_418M (https://huggingface.co/facebook/m2m100_418M, https://arxiv.org/abs/2010.11125)

### Parameters
Here is the list of training parameters:
* *mode*: train/test/generate
* *epochs*: the number of epochs
* *batch_size*: the batch size
* *test_batch_size*: the test batch size
* *max_source_length, max_target_length, min_target_length*: the max lengths of source and target
* *model_path*: the path of the trained model
* *train_path, val_path, test_path*: URLs of training, validation, and test files
* *source_prefix*: helpful for T5 models. 
* *source_column, target_column*: specify the fields of input and output in the training, validation, and test sets
* *decode_pred*: decode predictions back to Vietnamese (see utils.py)

### google-t5/t5-small
#### With prefix
python seq2seq.py --mode "train" --model_name "google-t5/t5-small" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "summarize: " --source_column "source" --target_column "target_encoded"

python seq2seq.py --mode "test" --model_name "google-t5/t5-small" --model_path "google-t5_t5-small\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target" --decode_pred 1

#### Without prefix
python seq2seq.py --mode "train" --model_name "google-t5/t5-small" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "" --source_column "source" --target_column "target_encoded"

python seq2seq.py --mode "test" --model_name "google-t5/t5-small" --model_path "google-t5_t5-small\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target" --decode_pred 1


### google-t5/t5-base
#### With prefix
python seq2seq.py --mode "train" --model_name "google-t5/t5-base" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "summarize: " --source_column "source" --target_column "target_encoded"

python seq2seq.py --mode "test" --model_name "google-t5/t5-base" --model_path "google-t5_t5-base\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target" --decode_pred 1

#### Without prefix
python seq2seq.py --mode "train" --model_name "google-t5/t5-base" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "" --source_column "source" --target_column "target_encoded"

python seq2seq.py --mode "test" --model_name "google-t5/t5-base" --model_path "google-t5_t5-base\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target" --decode_pred 1

### facebook/bart-base
python seq2seq.py --mode "train" --model_name "facebook/bart-base" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "" --source_column "source" --target_column "target_encoded"

python seq2seq.py --mode "test" --model_name "facebook/bart-base" --model_path "facebook_bart-base\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target" --decode_pred 1

### Helsinki-NLP/opus-mt-en-vi
python seq2seq.py --mode "train" --model_name "Helsinki-NLP/opus-mt-en-vi" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "" --source_column "source" --target_column "target"

python seq2seq.py --mode "test" --model_name "Helsinki-NLP/opus-mt-en-vi" --model_path "opus-mt-en-vi\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target"
 
### VietAI/envit5-translation

python seq2seq.py --mode "train" --model_name "VietAI/envit5-translation" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "" --source_column "source" --target_column "target"

python seq2seq.py --mode "test" --model_name "VietAI/envit5-translation" --model_path "envit5-translation\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target"

### facebook/m2m100_418M
python seq2seq.py --mode "train" --model_name "facebook/m2m100_418M" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "" --source_column "source" --target_column "target"

python seq2seq.py --mode "test" --model_name "facebook/m2m100_418M" --model_path "m2m100_418M\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target"

### facebook/mbart-large-50 (bad results)

python seq2seq.py --mode "train" --model_name "facebook/mbart-large-50" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "" --source_column "source" --target_column "target"

python seq2seq.py --mode "test" --model_name "facebook/mbart-large-50" --model_path "google_mt5-base\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "" --source_column "source" --target_column "target"

### google/mt5-base (worst results)

python seq2seq.py --mode "train" --model_name "google/mt5-base" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 3 --batch_size 4 --max_source_length 32 --source_prefix "summarize: " --source_column "source" --target_column "target"

python seq2seq.py --mode "test" --model_name "google/mt5-base" --model_path "google_mt5-base\checkpoint-xxx" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 32 --min_target_length 1 --source_prefix "summarize: " --source_column "source" --target_column "target"

# 6. Contact
* tahoangthang@gmail.com
* https://www.tahoangthang.com
