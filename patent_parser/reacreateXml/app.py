def convertFile(): 
	path = "../data_set/"
	file_name = "CLMS_US_0059"
	input_name = file_name + ".OUT"
	output_name = file_name + ".xml"

	f = open(path + input_name,"r+")
	d = f.readlines()
	filter_file = open(path + output_name, 'w')

	filter_file.write('<root>')  
	for line in d:
		#if len(line) > 100:
		if 'PATN' not in line and 'RN' not in line :
			line = line.replace('<CLMS>', '')
			filter_file.write(line)  
	filter_file.write('</root>')  
	f.close()
	filter_file.close()

convertFile()


