import numpy as np
import matplotlib.pyplot as plt

from file_io import *

import string
PUNCTS = string.punctuation + '”“¿?.�✔✅⤵➕➖⛔✍⃣.-'
PUNCTS = list(PUNCTS) # punctuations

import spacy
nlp_en = spacy.load('en_core_web_md')
STOPWORDS_EN = nlp_en.Defaults.stop_words

from spacy.lang.vi import Vietnamese # https://github.com/trungtv/vi_spacy
nlp_vi = Vietnamese()

STOPWORDS_VI = read_list_from_text_file('dataset/vietnamese-stopwords.txt')

def count_vocab_size(input_file = 'dataset/collected_data.json', word_type = 'insen'):

    """
        word_type: insen (insensitive), sen (sensitive)
    """
    dataset = read_list_from_jsonl_file(input_file)
    vocab_en_dict = {}
    vocab_vi_dict = {}

    for item in dataset:

        source = item['source']
        doc_en = nlp_en(source)
        for token in doc_en:
            if (word_type == 'sen'):
                word = token.text.strip()
            else:
                word = token.text.strip().lower()
            if (word == '' or word in PUNCTS): continue
            
            if (word not in vocab_en_dict):
                vocab_en_dict[word] = 1
            else:
                vocab_en_dict[word] += 1
        
        target = item['target']
        doc_vi = nlp_vi(target)
        for token in doc_vi:
            if (word_type == 'sen'):
                word = token.text.strip()
            else:
                word = token.text.strip().lower()
            if (word == '' or word in PUNCTS): continue
            
            if (word not in vocab_vi_dict):
                vocab_vi_dict[word] = 1
            else:
                vocab_vi_dict[word] += 1
                
    print('***************************')
    print('vocab_en_dict: ', len(vocab_en_dict))
    vocab_en_dict = dict(sorted(vocab_en_dict.items(), key=lambda item: item[1], reverse = True))
    write_list_to_json_file('dataset/vocab_en.json', vocab_en_dict, file_access = 'w')
    
    vocab_en_dict = dict(sorted(vocab_en_dict.items(), key=lambda item: item[1], reverse = False))
    print('most rare words: ', list(vocab_en_dict.items())[0:20])
    
    print('------')
    print('vocab_vi_dict: ', len(vocab_vi_dict))
    vocab_vi_dict = dict(sorted(vocab_vi_dict.items(), key=lambda item: item[1], reverse = True))
    write_list_to_json_file('dataset/vocab_vi.json', vocab_vi_dict, file_access = 'w')
    
    vocab_vi_dict = dict(sorted(vocab_vi_dict.items(), key=lambda item: item[1], reverse = False))
    print('most rare words: ', list(vocab_vi_dict.items())[0:20])
    
    print('***************************')


def show_top_vocab_plot(input_file = 'dataset/collected_data.json', x_label = '', y_label = 'Frequency', word_type = 'insen'):
    
    dataset = read_list_from_jsonl_file(input_file)
    vocab_en_dict = {}
    vocab_vi_dict = {}
    
    for item in dataset:
        source = item['source']
        doc_en = nlp_en(source)
        for token in doc_en:
            if (word_type == 'sen'):
                word = token.text.strip()
            else:
                word = token.text.strip().lower()
            
            if (word == '' or word in PUNCTS or word in STOPWORDS_EN): continue
            
            if (word not in vocab_en_dict):
                vocab_en_dict[word] = 1
            else:
                vocab_en_dict[word] += 1
        
        target = item['target']
        doc_vi = nlp_vi(target)
        for token in doc_vi:
            if (word_type == 'sen'):
                word = token.text.strip()
            else:
                word = token.text.strip().lower()
            if (word == '' or word in PUNCTS or word in STOPWORDS_VI): continue
            
            if (word not in vocab_vi_dict):
                vocab_vi_dict[word] = 1
            else:
                vocab_vi_dict[word] += 1
                
    vocab_en_dict = dict(sorted(vocab_en_dict.items(), key=lambda item: item[1], reverse = True))
    vocab_vi_dict = dict(sorted(vocab_vi_dict.items(), key=lambda item: item[1], reverse = True))
    
    label_list, value_list = [], []
    i = 1
    for k, v in vocab_en_dict.items():
        label_list.append(k)
        value_list.append(v)
        i = i + 1
        if (i > 20): break
    
    label_list = np.array(label_list)
    x_list = np.array(range(1, len(label_list) + 1))
    
    
    label_list2, value_list2 = [], []
    i = 1
    for k, v in vocab_vi_dict.items():
        label_list2.append(k)
        value_list2.append(v)
        i = i + 1
        if (i > 20): break
        
    label_list2 = np.array(label_list2)
    plt.rcParams.update({'font.size': 10})

    # show 2 plots
    plt.bar(x_list - 0.2, value_list, 0.4, label = 'English')
    plt.bar(x_list + 0.2, value_list2, 0.4, label =  'Vietnamese')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i in range(len(label_list)):
        plt.annotate(label_list[i],
                     xy=(x_list[i] - 0.2,value_list[i] + 0.1), ha='center', va='bottom', rotation = 65)
        
        plt.annotate(label_list2[i],
                     xy=(x_list[i] + 0.2,value_list2[i]+ 0.1), ha='center', va='bottom', rotation = 65)
                     
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.xlim([0, 21])
    #plt.figure(figsize=(8,5))
    plt.legend()
    plt.show()
    
    

def show_length_plot(source_dict, target_dict, x_label = 'Text length', y_label = 'Number of texts', bar_label1 = 'Source', bar_label2 = 'Target'):

    # 661: 1, len 661 appear 1 time
    
    label_list, value_list = [], []
    for k, v in source_dict.items():
        label_list.append(k)
        value_list.append(v)

    label_list2, value_list2 = [], []
    print('value_list2: ', value_list2)
    for k, v in target_dict.items():
        label_list2.append(k)
        value_list2.append(v)

    print('len2: ', len(label_list2))
    print('len1: ', len(label_list))

    for i in range(len(label_list2), len(label_list)):
        print(i)
        label_list2.append(i)
        value_list2.append(0)

    print('value_list2: ', value_list2, len(value_list2))
    print('value_list: ', value_list, len(value_list))

    print('len2: ', len(label_list2))
    print('len1: ', len(label_list))
    
    label_list = np.array(label_list)
    label_list2 = np.array(label_list2)


    plt.rcParams.update({'font.size': 12})
    
    plt.bar(label_list - 0.2, value_list, label = bar_label1)
    plt.bar(label_list2 + 0.2, value_list2, label =  bar_label2)

    
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    '''for i in range(len(label_list)):
        plt.annotate(str(value_list[i]) + ' (' + str(round(percent_list[i]*100, 2)) + ' %)',
                     xy=(label_list[i],value_list[i]), ha='center', va='bottom')'''
    plt.grid(linestyle = '--', linewidth = 0.5)
    #plt.xlim([0, 20])
    #plt.figure(figsize=(8,5))
    plt.legend()
    plt.show()
    
def length_distribution(input_file = 'dataset/collected_data.json'):
    
    dataset = read_list_from_jsonl_file(input_file)
    target_dict = {}
    source_dict = {}  
    
    for item in dataset:
    
        doc_source = nlp_en(item['source'])
        doc_source = [token.text.strip() for token in doc_source if token.text.strip() not in PUNCTS and token.text.strip() != '']
        if (len(doc_source) not in source_dict): source_dict[len(doc_source)] = 1
        else: source_dict[len(doc_source)] += 1
        
        doc_target = nlp_vi(item['target'])
        doc_target = [token.text.strip() for token in doc_target if token.text.strip() not in PUNCTS and token.text.strip() != '']
        if (len(doc_target) not in target_dict): target_dict[len(doc_target)] = 1
        else: target_dict[len(doc_target)] += 1
        
    source_dict = dict(sorted(source_dict.items(), key = lambda x:x[0], reverse = True))
    target_dict = dict(sorted(target_dict.items(), key = lambda x:x[0], reverse = True))
    
    show_length_plot(source_dict, target_dict, x_label = 'Text length', y_label = 'Number of texts')
    

#...............................................................................            
if __name__ == "__main__":
    count_vocab_size(word_type = 'sen')
    count_vocab_size(word_type = 'insen')
    show_top_vocab_plot()
    length_distribution()
    