from aiohttp import web
import socketio
import threading
import asyncio
from audio_analysis import audio_analyzer


class ServerOperation:
    """
        Server operation for handling voice data over web sockets using aiohttp and socket.io.

        Learn more about socket.io with Python: https://python-socketio.readthedocs.io/en/latest/
    """
    def __init__(self):
        self.sio = socketio.AsyncServer(cors_allowed_origins="*")
        self.app = web.Application()
        self.sio.attach(self.app)

        self.app.router.add_get('/', self.index)

        @self.sio.event
        def connect(sid, environ):
            print("connect ", sid)

        @self.sio.event
        async def chat_message(sid, data):
            print("message ", data)

        @self.sio.on('get_result')
        async def socket_get_result(sid, data):
            asyncio.create_task(self.process_audio_and_respond(sid))

        @self.sio.event
        def disconnect(sid):
            print('disconnect ', sid)

    async def index(self, request):
        text = await audio_analyzer.process_audio()
        return web.Response(text=text, content_type='text/plain')

    async def process_audio_and_respond(self, sid):
        text = await audio_analyzer.process_audio()
        await self.sio.emit('response', text, room=sid)

    def run_server(self):
        if threading.current_thread() is threading.main_thread():
            web.run_app(self.app, port=5678)
        else:
            print("=> The ship must be launched in the main thread! <=")


server_operation = ServerOperation()
