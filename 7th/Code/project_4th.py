import matplotlib.pyplot as plt
import csv
import re

x_jio = []
y_jio = []

with open('csvname_jio.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
    	try:
    		#print(row)
    		x_jio.append(int(row[0]))
    		s=[float(s) for s in re.findall(r'[-+]?\d*\.\d+|\d+',row[2])]
    		#print(s)
    		y_jio.append(s[0])
    	except:
    		continue

x_airtel = []
y_airtel = []

with open('csvname_airtel.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
    	try:
    		#print(row[0])
    		x_airtel.append(int(row[0]))
    		s=[float(s) for s in re.findall(r'[-+]?\d*\.\d+|\d+',row[2])]
    		print(s)
    		y_airtel.append(s[0])
    	except:
    		continue
    # print(x_airtel)
    # print(y_airtel)

plt.plot(x_jio,y_jio, label='Jio')
plt.plot(x_airtel,y_airtel, label='Airtel')
plt.xlabel('Number of Days')
plt.ylabel('Kbps')
plt.title('Comparison between two operator of Uploading speed')
plt.legend()
plt.show()