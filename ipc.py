#!/usr/bin/python
"""
IPC
"""

import socketio
import eventlet
import logger as d
import func as fx

## socket defaults
sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': '../index.html'}
})

# events
@sio.on('connect')
def connect(sid, environ):
    d.success('Client socket opened => {}'.format(sid))

@sio.on('disconnect')
def disconnect(sid):
    d.error('Client socket closed => {}'.format(sid))


# daemon
if __name__ == '__main__':
    d.default('[+] IPC running: {}:{}'.format(fx.host_ip, fx.host_port))
    eventlet.wsgi.server(eventlet.listen((fx.host_ip, fx.host_port)), app)
