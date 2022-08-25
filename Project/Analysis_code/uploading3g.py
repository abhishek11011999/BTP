import matplotlib.pyplot as plt
import csv
import re
from io import StringIO

x_airtel4g = []
y_airtel4g = []

with open('csvname_airtel.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        try:
            print(row)
            x_airtel4g.append(int(row[0]))
            s=[float(s) for s in re.findall(r'[-+]?\d*\.\d+|\d+',row[2])]
    		#print(s)
            y_airtel4g.append(s[0])
        except:
            continue

x_airtel3g = []
y_airtel3g = []

with open('csvname_airtel3g.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        print(row)
        x_airtel3g.append(int(row[0]))
        s=[float(s) for s in re.findall(r'[-+]?\d*\.\d+|\d+',row[2])]
        #print(s)
        y_airtel3g.append(s[0])
    # print(x_airtel)
    # print(y_airtel)

plt.plot(x_airtel4g,y_airtel4g, label='Airtel4G')
plt.plot(x_airtel3g,y_airtel3g, label='Airtel3G')
plt.xlabel('Number of Days')
plt.ylabel('Kbps')
plt.title('Comparison of upload speed between same operator of 4G and 3G')
plt.legend()
plt.show()