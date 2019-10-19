# -*- coding:utf-8 -*-
import xml.etree.ElementTree as ET

# reference : https://stackabuse.com/reading-and-writing-xml-files-in-python/

def parseXml49():
	path = "../data_set/"
	file_name = "CLMS_US_0049"
	input_name = file_name + ".xml"
	output_name = file_name + "_parsed.xml"

	tree = ET.parse(path + input_name)  # 파일을 이용한 파싱
	root = tree.getroot()

	# create the file structure
	data = ET.Element('data')

	for CLAIMs in root:  # tag : CLAIMs
		for CLAIM in CLAIMs:  # tag : CLAIM	
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

def parseXml59():
	path = "../data_set/"
	file_name = "CLMS_US_0059"
	input_name = file_name + ".xml"
	output_name = file_name + "_parsed.xml"

	tree = ET.parse(path + input_name)
	root = tree.getroot()

	# create the file structure
	data = ET.Element('data')

	for CLAIMs in root:  # tag : CLAIMs
		for CLAIM in CLAIMs:  # tag : CLAIM
			for claim_text in CLAIM:  # tag : claim-text
				if 'claim-text' not in claim_text.tag:
					break
				for claim_ref in claim_text:  # tag : claim-ref
					if 'claim-ref' in claim_ref.tag:
						item = ET.SubElement(data, 'item')
						
						PDAT1 = ET.SubElement(item, 'PDAT1')
						PDAT1.text = claim_text.text

						CLREF = ET.SubElement(item, 'CLREF')
						CLREF.set('CLREF',claim_ref.attrib['idref'])

						PDAT2 = ET.SubElement(item, 'PDAT2')
						PDAT2.text = claim_ref.tail
						

	filter_file = open(path + output_name, 'w')							
	mydata = ET.tostring(data)
	filter_file.write(mydata)
	filter_file.close()

parseXml59()
