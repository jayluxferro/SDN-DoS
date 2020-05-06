"""
Generate results
"""
import db

header = ['Protocol', 'Time', 'Model']

# detection
handler = open('detection_result.csv', 'w')

data = ','.join(header) + '\n'

for d in db.fetchTable('defense'):
    data += '{}, {}, {}\n'.format(d['protocol'], d['time'], d['model'])
handler.write(data)
handler.close()




# mitigation
handler = open('mitigation_result.csv', 'w')

data = ','.join(header) + '\n'

for d in db.fetchTable('detection'):
    data += '{}, {}, {}\n'.format(d['protocol'], d['time'], d['model'])
handler.write(data)
handler.close()
