import re

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

def encode_vi(txt):
    
    new_txt = ''
    for x in txt:
        temp = x
        if (x in VICODE):
            #print(x, "@" + str(VICODE.index(x)))
            temp = "@" + str(VICODE.index(x))
        new_txt += temp
    
    return new_txt


def decode_vi(txt):

    re_lst = re.findall('@[0-9]+', txt)
    for x in re_lst:
        txt = txt.replace(x, VICODE[int(x[1:])])
    return txt


'''
text = 'Xin chào các bạn ở Việt Nam'
encode_text = encode_vi(text)
print('encode_text: ', encode_text)
print('encode_text: ', decode_vi(encode_text))'''