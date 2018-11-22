from __future__ import unicode_literals
from threading import Timer
from wxpy import *
import random
import datetime

bot = Bot()
bot = Bot(console_qr=2,cache_path="botoo.pkl")

for i in bot.groups():
    print(i)
traget_fri = ensure_one(bot.friends().search('王贤敏'))
my_group = bot.groups().search('abc')
target_group = ensure_one(my_group)
print("find")
@bot.register(target_group)
def forward_sharing_message(msg):
    print(msg)
    msg.forward(traget_fri)
    print("Sent!")

embed()