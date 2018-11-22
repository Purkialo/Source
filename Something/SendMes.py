from __future__ import unicode_literals
from threading import Timer
#from wxpy import *
import requests
import random
import datetime

#bot = Bot()
# linux执行登陆请调用下面的这句
#bot = Bot(console_qr=2,cache_path="botoo.pkl")
def get_news():
    """获取金山词霸每日一句，英文和翻译"""
    url = "http://open.iciba.com/dsapi/"
    r = requests.get(url)
    content = r.json()['content']
    note = r.json()['note']
    return content, note

def send_news():
    while(1):
        t = datetime.datetime.now().strftime('%H:%M')
        if(t == "05:20"):
            try:
                contents = get_news()
                # 你朋友的微信名称，不是备注，也不是微信帐号。
                my_friend = bot.friends().search('Camille')[0]
                my_friend.send(contents[0])
                # my_friend.send(contents[1])
                # my_friend.send(u'遇见你不容易，错过了会很可惜。')
                my_friend.send(u"早安,测试代码")
                break
            except:

                # 你的微信名称，不是微信帐号。
                my_friend = bot.friends().search('WhitePanda')[0]
                my_friend.send(u"今天消息发送失败了")

if __name__ == "__main__":
    send_news()
