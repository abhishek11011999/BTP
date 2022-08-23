import matplotlib.pyplot as plt
import numpy as np
import csv
import re

signal_data_jio=[]
signal_data_airtel=[]


with open('csvname_jio.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
    	try:
    		s=[i for i in row[4].split()]
    		signal_data_jio.append(s[3])
    		#print(signal_data)
    		#y_jio.append(s[0])
    	except:
    		continue

total_signal=0
for i in range(0,len(signal_data_jio)):
	total_signal+=int(signal_data_jio[i])

avg_signal_strength_jio=abs(total_signal/len(signal_data_jio))

print(total_signal)
print(len(signal_data_jio))
print(avg_signal_strength_jio)


with open('csvname_airtel.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
    	try:
    		s=[i for i in row[4].split()]
    		signal_data_airtel.append(s[1])
    		#print(signal_data)
    		#y_jio.append(s[0])
    	except:
    		continue

total_signal=0
for i in range(0,len(signal_data_airtel)):
	total_signal+=int(signal_data_airtel[i])

avg_signal_strength_airtel=abs(total_signal/len(signal_data_airtel))
print(total_signal)
print(len(signal_data_airtel))
print(avg_signal_strength_airtel)


# xdata=("Jio","Airtel")
# x_data=np.arange(len(xdata))
# y_data=[avg_signal_strength_jio,avg_signal_strength_airtel]

# plt.bar(x_data, y_data, align='center', alpha=0.5, width = 0.1)
# plt.xticks(x_data,xdata)
# plt.ylabel('Signal strength (In negative)')
# plt.title('comaparing signal strength')

# plt.show()

n_groups = 4
means_frank = (avg_signal_strength_jio,avg_signal_strength_airtel,0,0)
means_guido = (avg_signal_strength_jio-20,avg_signal_strength_airtel-18,0,0)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.5

rects1 = plt.bar(index, means_frank, bar_width,
alpha=opacity,
color='b',
label='Mozilla Firefox',
align='edge')

rects2 = plt.bar(index + bar_width, means_guido, bar_width,
alpha=opacity,
color='g',
label='Google Chrome',
align='edge')

plt.xlabel('Operator')
plt.ylabel('Signal Strength (In Negative)')
plt.title('comaparing Signal Strength')
plt.xticks(index + bar_width, ('Jio', 'Airtel'))
plt.legend()

plt.tight_layout()
plt.show()
