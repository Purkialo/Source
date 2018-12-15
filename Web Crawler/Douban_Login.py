# -*- coding: utf-8 -*-
"""
Created on Thu DEC 11 2018
@author: Purkialo
"""
import time
import numpy as np
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image

headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Referer':'https://accounts.douban.com/login?alias=&redir=https%3A%2F%2Fwww.douban.com%2F&source=index_nav&error=1001'
}

session = requests.Session()
session.headers.update(headers)

username = "wildpanda@yeah.net"
password = "chudouban123"

url = 'https://accounts.douban.com/login'

data = {
    'source':'None',
    'redir':'https://www.douban.com/',
    'form_email':username,
    'form_password':password,
    'remember':'on',
    'login':'登录',
}

# try:
#     html = requests.get(url)
#     soup = BeautifulSoup(html.text,'lxml')
#     caprcha_link = soup.select('#captcha_image')[0]['src']
#     caprcha_id = soup.select('div.captcha_block > input')[1]['value']
# except:
#     caprcha_id = False
# if caprcha_id:
#     img_html = session.get(caprcha_link)
#     with open('caprcha.jpg','wb') as f:
#         f.write(img_html.content)
#     try:
#         im = Image.open('caprcha.jpg')
#         im.show()
#         im.close()
#     except:
#         print('打开错误')
#     caprcha = input('请输入验证码：')
#     data['captcha-id'] = caprcha_id
#     data['captcha-solution'] = caprcha

# session.post(url,data=data,headers=headers)
# print(session.cookies.items())

def book_spider(book_tag,session):
    page_num=0
    book_list=[]
    book_id = 0
    while(1):
        url='http://www.douban.com/tag/'+book_tag+'/book?start='+str(page_num*15)
        print(url)
        time.sleep(np.random.rand()*3)
        try:
            req = session.get(url)
        except:
            print("Get url error!  Over!")
            return
        source_code = req.text
        
        soup = BeautifulSoup(source_code,"html.parser")
        list_soup = soup.find('div', {'class': 'mod book-list'})
        for book_info in list_soup.findAll('dd'):
            title = book_info.find('a', {'class':'title'}).string.strip()
            desc = book_info.find('div', {'class':'desc'}).string.strip()
            desc_list = desc.split('/')
            book_url = book_info.find('a', {'class':'title'}).get('href')
            try:
                author_info = '作者/译者： ' + '/'.join(desc_list[0:-3])
            except:
                author_info ='作者/译者： 暂无'
            try:
                pub_info = '出版信息： ' + '/'.join(desc_list[-3:])
            except:
                pub_info = '出版信息： 暂无'
            try:
                rating = book_info.find('span', {'class':'rating_nums'}).string.strip()
            except:
                rating='0.0'
            try:
                print(book_url)
                intro, aut_intro, short_review_url, long_review_url = get_intro(session,book_url)
                get_review(session, short_review_url, long_review_url, book_id)
            except:
                intro = ""
                aut_intro = ""
            
            book_list.append([book_id, title, rating, author_info, pub_info, intro, aut_intro])
            book_list_buffer = np.array(book_list)
            book_list_buffer = book_list_buffer.reshape((-1,7))
            csvfile_book_list = pd.DataFrame(book_list_buffer,columns=["1","1","1","1","1","1","1"])
            #将dataframe写入文件
            csvfile_book_list_path = "bookname.csv"
            csvfile_book_list.to_csv(csvfile_book_list_path,encoding="utf_8_sig",index=False)
            print("Book info written!\n")
            book_id += 1
        page_num+=1
        print ('Downloading Information From Page %d' % page_num)

def get_intro(session,url):
    #try:
    time.sleep(np.random.rand()*5)
    req = session.get(url)
    source_code = req.text
    soup = BeautifulSoup(source_code,"html.parser")
    all_intro = soup.find_all('div',{'class':'intro'})
    short_review_url = url[:-13] + "comments"
    long_review_url = url[:-13] + "reviews"
    if(len(all_intro) > 3):
        intro = all_intro[1].get_text().strip()
        aut_intro = all_intro[3].get_text().strip()
    elif(len(all_intro) > 2):
        intro = all_intro[1].get_text().strip()
        aut_intro = all_intro[2].get_text().strip()
    else:
        intro = all_intro[0].get_text().strip()
        aut_intro = all_intro[1].get_text().strip()
    return intro, aut_intro, short_review_url, long_review_url

def get_review(session, short_review_url,long_review_url,book_id):
    print("Begin to get comments and reviews!")
    time.sleep(np.random.rand())
    time.sleep(np.random.rand()*2)
    req = session.get(short_review_url)
    source_code = req.text
    soup = BeautifulSoup(source_code,"html.parser")
    comment_num = soup.find('span',{'id':'total-comments'}).get_text()
    comment_num = re.sub(r"\D", "", comment_num)
    comment_num = int(float(comment_num) / 20) + 1
    print("comments_num : %d"%comment_num)

    time.sleep(np.random.rand()*3)
    req = session.get(long_review_url)
    source_code = req.text
    soup = BeautifulSoup(source_code,"html.parser")
    review_num = soup.find('div',{'id':'content'}).h1.get_text()
    try:
        review_num = review_num.split('(',1)[1]
        review_num = re.sub(r"\D", "", review_num)
        review_num = int(float(review_num) / 20) + 1
    except:
        review_num = 1
    print("reviews_num : %d"%review_num)
    
    comments = []
    reviews = []
    page_num = 0
    while(1):
        if(page_num != 0):
            comment_url = short_review_url + '/hot?p='+str(page_num)
        else:
            comment_url = short_review_url
        if(page_num%10 == 0):
            print(comment_url)
            print("process comment no %d, page %d, len: %d"%(book_id,page_num + 1, len(comments)))
        if(page_num%30 == 0):
            save_to_file(comments,reviews,book_id)
        if(page_num > comment_num):
            break
        page_num += 1

        time.sleep(np.random.rand()*2)
        req = session.get(comment_url)
        source_code = req.text
        soup = BeautifulSoup(source_code,"html.parser")
        all_comment = soup.find_all('span',{'class':'short'})
        for i in range(len(all_comment)):
            comments.append(all_comment[i].get_text())
    page_num = 0
    while(1):
        if(page_num != 0):
            review_url = long_review_url + '?start='+str(page_num * 20)
        else:
            review_url = long_review_url
        if(page_num > review_num):
            break
        if(page_num%2 == 0):
            print("process review no %d, page %d, len: %d"%(book_id,page_num + 1, len(reviews)))
        if(page_num%2 == 0):
            save_to_file(comments,reviews,book_id)

        page_num += 1

        time.sleep(np.random.rand()*3)
        req = session.get(review_url)
        source_code = req.text
        soup = BeautifulSoup(source_code,"html.parser")
        all_review = soup.find_all('div',{'class':'review-short'})
        for i in range(len(all_review)):
            full_review_id = all_review[i].get('data-rid')
            full_review_url = "https://book.douban.com/j/review/" + str(full_review_id) + "/full"
            #print(full_review_url)
            time.sleep(np.random.rand())
            rev_req = session.get(full_review_url)
            rev_source_code = rev_req.text
            rev_soup = BeautifulSoup(rev_source_code,"html.parser")
            review = rev_soup.find('div').get_text()
            tag = re.compile(r'<[^>]+>',re.S)
            review = tag.sub('',review)
            tag = re.compile(r'\\n',re.S)
            review = tag.sub('',review)
            tag = re.compile(r'  ',re.S)
            review = tag.sub('',review)
            review = review.split("vote_script")[0][:-3]
            reviews.append(review)
    save_to_file(comments,reviews,book_id)

def save_to_file(comments,reviews,book_id):
    comments = np.array(comments).reshape((-1,1))
    reviews = np.array(reviews).reshape((-1,1))
    csvfile_comments = pd.DataFrame(comments,columns=["短评论"])
    csvfile_reviews = pd.DataFrame(reviews,columns=["长评论"])
    print(csvfile_comments.shape)
    print(csvfile_reviews.shape)
    #将dataframe写入文件
    csv_comments_path = "%d_short.csv"%(book_id)
    try:
        csvfile_comments.to_csv(csv_comments_path,encoding="utf_8_sig",index=False)
        print("Comments written!")
    except:
        try:
            csvfile_comments.to_csv(csv_comments_path,encoding="utf_8_sig",index=False)
            print("Comments written!")
        except:
            print("Comments save Error!")

    csv_reviews_path = "%d_long.csv"%(book_id)
    try:
        csvfile_reviews.to_csv(csv_reviews_path,encoding="utf_8_sig",index=False)
        print("Reviews written!")
    except:
        try:
            csvfile_reviews.to_csv(csv_reviews_path,encoding="utf_8_sig",index=False)
            print("Reviews written!")
        except:
            print("Reviews save Error!")

#book_tag_lists = ['历史']
#book_tag_lists = ['爱情']
#book_tag_lists = ['哲学']
#book_tag_lists = ['计算机']
#book_tag_lists = ['传记']
#book_tag_lists = ['教育']
#book_tag_lists = ['商业']
#book_tag_lists = ['投资']
#book_tag_lists = ['名著']
#book_tag_lists = ['文学']
book_tag_lists = ['科幻']
#book_tag_lists = ['宗教']
book_spider(book_tag_lists[0],session)