# -*- coding:utf-8 -*-
import xml.etree.ElementTree as ET

# reference : https://stackabuse.com/reading-and-writing-xml-files-in-python/

def parseXml():
	path = "../data_set/"
	file_name = "CLMS_US_0049"
	input_name = file_name + ".xml"
	output_name = file_name + "_parsed.xml"

	tree = ET.parse(path + input_name)  # 파일을 이용한 파싱
	root = tree.getroot()

	# create the file structure
	data = ET.Element('data')

	for child in root:  # tag : CLAIMs
		for CLAIM in child:  # tag : CLAIM
			# print(CLAIM.tag, CLAIM.text)
			for PARA in CLAIM:  # tag : PARA, CLMSTEP
				if 'PARA' not in PARA.tag:
					break
				# print(PARA.tag, PARA.text)
				for PTEXT in PARA:  # tag : PARA -> PTEXT
					if 'PTEXT' not in PTEXT.tag:
						break
					item = ET.SubElement(data, 'item')
					checkRef = False
					flag = True

					for ele in PTEXT:  # tag : ele -> PDAT, CLREF, PDAT
						if 'CLREF' in ele.tag :
							CLREF = ET.SubElement(item, 'CLREF')
							CLREF.set('CLREF',ele.attrib['ID'])
							checkRef = True
						if 'PDAT' in ele.tag :
							if(flag) :
								PDAT1 = ET.SubElement(item, 'PDAT1')
								PDAT1.text = ele.text
								flag = False
							else : 
								PDAT2 = ET.SubElement(item, 'PDAT2')
								PDAT2.text = ele.text
					if checkRef == False:
						data.remove(item)

	filter_file = open(path + output_name, 'w')							
	mydata = ET.tostring(data)
	filter_file.write(mydata)
	filter_file.close()

parseXml()
