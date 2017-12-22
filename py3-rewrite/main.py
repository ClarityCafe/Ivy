import asyncio

from aiohttp import web
from chatbot import Chatbot
from concurrent.futures import ThreadPoolExecutor

app = web.Application()
bot = Chatbot()
loop = asyncio.get_event_loop()
executor = ThreadPoolExecutor(max_workers=20)

async def handle_request(req):
    body = await req.text()

    if not body:
        return web.Response(status=400, text='{"error": "No body.", "code": 0}', content_type='application/json')

    resp = await loop.run_in_executor(executor, bot.chat, body)

    return web.Response(text=f'{{"text": {resp}}}', content_type='image/png')

app.router.add_post('/chat', handle_request)
web.run_app(app)