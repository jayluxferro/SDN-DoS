"""
Generate data
"""
import db
import func as fx

file_name = 'data.csv'

header = fx.header
file_handler = open(file_name, 'w')
d = ','.join(header) + '\n'

for data in db.fetchTable('data'):
    d += '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(fx.formatIP(data['source_ip']), fx.formatIP(data['destination_ip']), fx.formatProtocol(data['protocol']), fx.formatSwitch(data['switch_mac']), int(data['p1']), int(data['p2']), int(data['p3']), int(data['p4']), int(data['p1_rx_packets']), int(data['p1_tx_packets']), int(data['p1_rx_bytes']), int(data['p1_tx_bytes']), int(data['p2_rx_packets']), int(data['p2_tx_packets']), int(data['p2_rx_bytes']), int(data['p2_tx_bytes']), int(data['p3_rx_packets']), int(data['p3_tx_packets']), int(data['p3_rx_bytes']), int(data['p3_tx_bytes']), int(data['p4_rx_packets']), int(data['p4_tx_packets']), int(data['p4_rx_bytes']), int(data['p4_tx_bytes']), int(data['label']))

file_handler.write(d)
file_handler.close()
