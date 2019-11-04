# -*- coding:utf-8 -*-
import xml.etree.ElementTree as ET

# reference : https://stackabuse.com/reading-and-writing-xml-files-in-python/

def parseXml49():
    path = "../data_set/"
    file_name = "CLMS_US_0049"
    input_name = file_name + ".xml"
    output_name = file_name + "_w2v_parsed.xml"

    tree = ET.parse(path + input_name)
    root = tree.getroot()

    data = ET.Element('data')
    non_claim_text_in_ref_cnt = 0
    for CLAIMs in root:
        for CLAIM in CLAIMs:
            if 'CLAIM' not in CLAIM.tag :
                continue
            item = ET.SubElement(data,'item')
            num_tag = ET.SubElement(item,'no')
            text_tag = ET.SubElement(item,'text')
            ref_tag = ET.SubElement(item,'ref')
            isdep_tag = ET.SubElement(item,'isdep')
            if 'ID' in CLAIM.attrib:
                num_tag.text = CLAIM.attrib['ID']
            
            text = ''
            checkRef = False
            for PARA in CLAIM:
                if 'PDAT' in PARA.tag and PARA.text != None:
                    text += PARA.text

                for PTEXT in PARA:
                    if 'PDAT' in PTEXT.tag and PTEXT.text != None:
                        text += PTEXT.text

                    for CLREF in PTEXT:
                        if 'PDAT' in CLREF.tag and CLREF.text != None:
                            text += CLREF.text
                        if 'CLREF' in CLREF.tag :
                            checkRef = True
                            if 'ID' in CLREF.attrib:
                                ref_tag.text = CLREF.attrib['ID']

                        for PDAT in CLREF:
                            if 'PDAT' in PDAT.tag and PDAT.text != None:
                                text += PDAT.text
                                if checkRef == True and 'claim' not in PDAT.text:
                                    non_claim_text_in_ref_cnt+=1
                        
            if checkRef == False : 
                isdep_tag.text = '0'
                ref_tag.text = '0'
            else :
                isdep_tag.text = '1'
            idx = text.find('.')
            if idx != None and idx < 3 and idx > 0:
                text = text[idx+2: ]
            text_tag.text = text

    print('claims49 -> non_claim_text_in_ref_cnt : ', non_claim_text_in_ref_cnt)

    filter_file = open(path + output_name, 'w')					
    mydata = ET.tostring(data)
    filter_file.write(mydata)
    filter_file.close()


def parseXml59():
    path = "../data_set/"
    file_name = "CLMS_US_0059"
    input_name = file_name + ".xml"
    output_name = file_name + "_w2v_parsed.xml"

    tree = ET.parse(path + input_name)
    root = tree.getroot()

    data = ET.Element('data')
    non_claim_text_in_ref_cnt = 0

    for CLAIMs in root:
        for CLAIM in CLAIMs:
            if 'CLAIM' not in CLAIM.tag :
                continue
            item = ET.SubElement(data,'item')
            num_tag = ET.SubElement(item,'no')
            text_tag = ET.SubElement(item,'text')
            ref_tag = ET.SubElement(item,'ref')
            isdep_tag = ET.SubElement(item,'isdep')
            if 'ID' in CLAIM.attrib:
                num_tag.text = CLAIM.attrib['id']
            
            text = ''
            checkRef = False
            for tree1 in CLAIM:
                if 'claim-text' in tree1.tag and tree1.text != None:
                    text += tree1.text

                for tree2 in tree1:
                    if 'claim-text' in tree2.tag and tree2.text != None:
                        text += tree2.text
                    if 'claim-ref' in tree2.tag :
                        checkRef = True
                        if 'idref' in tree2.attrib:
                            ref_tag.text = tree2.attrib['idref']
                        if tree2.text != None:
                            text += tree2.text
                            if checkRef == True and 'claim' not in tree2.text:
                                non_claim_text_in_ref_cnt+=1
                        if tree2.tail != None:
                            text += tree2.tail
                    for tree3 in tree1:
                        if 'claim-text' in tree3.tag and tree3.text != None:
                            text += tree3.text

            if checkRef == False : 
                isdep_tag.text = '0'
                ref_tag.text = '0'
            else :
                isdep_tag.text = '1'
            idx = text.find('.')
            if idx != None and idx < 3 and idx > 0:
                text = text[idx+2: ]
            text_tag.text = text

    print('claims59 -> non_claim_text_in_ref_cnt : ', non_claim_text_in_ref_cnt)
        
    filter_file = open(path + output_name, 'w')					
    mydata = ET.tostring(data)
    filter_file.write(mydata)
    filter_file.close()

parseXml49()
parseXml59()
