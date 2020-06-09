import csv
import re

minimum_word_num = 10000
input_data = []

with open('input.csv', 'r') as raw:
    lines = raw.readlines()
line = csv.reader(lines)
input_data = line
line_count = 0
for data in line:
    line_count+=1
    if line_count == 1:
        continue
    attr_count = 0
    for attr in data:
        if attr_count == 1:
            words = attr.split(',')
            if len(words) < minimum_word_num:
                minimum_word_num = len(words)
        attr_count+=1

print('minimum_word_num: ')
print(minimum_word_num)

header = []
header.append('patn')
for i in range(1,minimum_word_num+1):
    header.append('c' + str(i))
header.append('i1')
header.append('i2')
header.append('i3')
header.append('i4')
line_count = 0

with open('ouput.csv', mode='w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=header)

    writer.writeheader()
    with open('input.csv', 'r') as raw:
        lines = raw.readlines()
    line = csv.reader(lines)
    for data in line:
        if line_count != 0:
            attr_count = 0
            thisdict = {}
            for attr in data:
                if attr_count == 0:
                    thisdict["patn"] = attr
                if attr_count == 1:
                    attr = attr.replace("[","")
                    attr = attr.replace("]","")
                    attr = attr.replace("'","") 
                    words = attr.split(',')
                    word_count = 1
                    for word in words:
                        
                        if word_count <= minimum_word_num:
                            thisdict['c' + str(word_count)] = word
                            word_count+=1
                if attr_count == 2:
                    indicators = attr.split('/')
                    indicator_count = 1
                    for indicator in indicators:
                        thisdict['i' + str(indicator_count)] = indicator
                        indicator_count+=1
                attr_count+=1
            writer.writerow(thisdict)
        line_count+=1      





