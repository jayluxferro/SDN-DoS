"""
Analysis
"""
import db
import numpy as np
import matplotlib.pyplot as plt

# detection
d_tcp = []
d_udp = []
d_icmp = []
file_handler = open('detection.csv', 'w')
data = ''
for _ in db.fetchTable('defense'): # i messed up the names when logging the data
    data += _['protocol']
    data += ',' + str(_['time']) + '\n'
    if _['protocol'] == 'tcp':
        d_tcp.append(_['time'])
    
    if _['protocol'] == 'udp':
        d_udp.append(_['time'])
    
    if _['protocol'] == 'icmp':
        d_icmp.append(_['time'])

file_handler.write(data)
file_handler.close()

# defense
df_tcp = []
df_udp = []
df_icmp = []
file_handler = open('mitigation.csv', 'w')
data = ''

for _ in db.fetchTable('detection'): # i messed up the names when logging the data
    data += _['protocol']
    data += ',' + str(_['time']) + '\n'
    if _['protocol'] == 'tcp':
        df_tcp.append(_['time'])
    
    if _['protocol'] == 'udp':
        df_udp.append(_['time'])
    
    if _['protocol'] == 'icmp':
        df_icmp.append(_['time'])

file_handler.write(data)
file_handler.close()

x = np.linspace(1, len(d_tcp), len(d_tcp))

plt.figure()
plt.plot(x, d_icmp, '-o', x, d_tcp, '-o', x, d_udp, '-o')
plt.xlabel('Attempts')
plt.ylabel('Detection Time (seconds)')
plt.title('Detection Time for ICMP, TCP and UDP Flood')
plt.legend(['ICMP', 'TCP', 'UDP'])
plt.show()

plt.figure()
plt.plot(x, df_icmp, '-o', x, df_tcp, '-o', x, df_udp, '-o')
plt.xlabel('Attempts')
plt.ylabel('Mitigation Time (seconds)')
plt.title('Mitigation Time for ICMP, TCP and UDP Flood')
plt.legend(['ICMP', 'TCP', 'UDP'])
plt.show()
