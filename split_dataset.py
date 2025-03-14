from file_io import *
import random

dataset = read_list_from_jsonl_file('dataset/collected_data.json')
random.shuffle(dataset) # shuffle dataset

train_set  = dataset[:int(len(dataset)*0.8)] # 80%

val_test_set =  dataset[int(len(dataset)*0.8):] # 20%
val_set =  val_test_set[0:int(len(val_test_set)*0.5)] # 50% of 20%
test_set = val_test_set[int(len(val_test_set)*0.5):] # 50% of 20%

# write files
write_list_to_jsonl_file('dataset/train.json', train_set, 'w')
write_list_to_jsonl_file('dataset/val.json', val_set, 'w')
write_list_to_jsonl_file('dataset/test.json', test_set, 'w')
