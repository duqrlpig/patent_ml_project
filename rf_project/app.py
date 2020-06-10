import csv
import re

# minimum_word_num = 10000
# input_data = []

# with open('input.csv', 'r') as raw:
#     lines = raw.readlines()
# line = csv.reader(lines)
# input_data = line
# line_count = 0
# for data in line:
#     line_count+=1
#     if line_count == 1:
#         continue
#     attr_count = 0
#     for attr in data:
#         if attr_count == 1:
#             words = attr.split(',')
#             if len(words) < minimum_word_num:
#                 minimum_word_num = len(words)
#         attr_count+=1

# print('minimum_word_num: ')
# print(minimum_word_num)

indicator_index = 3

indicators = ['FowardCitation', 'ipc', 'claim', 'family']

header = []
header.append('patn')
header.append('text')
header.append('indicator')
line_count = 0

with open('output_'+indicators[indicator_index]+'.csv', mode='w') as csv_file:
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
                    attr = attr.replace(","," ")
                    thisdict['text'] = attr
                if attr_count == 2:
                    indicators = attr.split('/')
                    indicator_count = 0
                    for indicator in indicators:
                        if indicator_count == indicator_index:
                            thisdict['indicator'] = indicator
                            indicator_count+=1
                attr_count+=1
            writer.writerow(thisdict)
        line_count+=1      





