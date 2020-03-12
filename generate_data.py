"""
Generate data
"""
import db

file_name = 'data.csv'

header=['Source IP', 'Destination IP', 'Protocol', 'Switch', 'P1', 'P2', 'P3', 'P4', 'P1 Rx Packet', 'P1 Tx Packet', 'P1 Rx Bytes', 'P1 Tx Bytes', 'P2 Rx Packet', 'P2 Tx Packet', 'P2 Rx Bytes', 'P2 Tx Bytes', 'P3 Rx Packet', 'P3 Tx Packet', 'P3 Rx Bytes', 'P3 Tx Bytes', 'P4 Rx Packet', 'P4 Tx Packet', 'P4 Rx Bytes', 'P4 Tx Bytes', 'Label']

file_handler = open(file_name, 'w')
d = ','.join(header) + '\n'

for data in db.fetchTable('data'):
    d += '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(data['source_ip'], data['destination_ip'], data['protocol'], data['switch_mac'], data['p1'], data['p2'], data['p3'], data['p4'], data['p1_rx_packets'], data['p1_tx_packets'], data['p1_rx_bytes'], data['p1_tx_bytes'], data['p2_rx_packets'], data['p2_tx_packets'], data['p2_rx_bytes'], data['p2_tx_bytes'], data['p3_rx_packets'], data['p3_tx_packets'], data['p3_rx_bytes'], data['p3_tx_bytes'], data['p4_rx_packets'], data['p4_tx_packets'], data['p4_rx_bytes'], data['p4_tx_bytes'], data['label'])

file_handler.write(d)
file_handler.close()
