#-*- coding: UTF-8 -*-
import time
import urllib
import urllib.request as urllib2
import numpy as np
import re
from bs4 import BeautifulSoup
import time
import numpy as np
import re
import pandas as pd

#Some User Agents
hds=[{'User-Agent':'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Trident/6.0)'},\
{'User-Agent':'Mozilla/5.0 (Windows NT 6.2) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.12 Safari/535.11'},\
{'User-Agent': 'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Trident/6.0)'}]
cookies=[{"cookie":"bid=0lGY_W7p8Q0; douban-fav-remind=1; __utmc=30149280; gr_user_id=a150c538-fdde-408d-8f81-fc6295fb61c8; _vwo_uuid_v2=D0FFEEFB02AA46A5A66F75BC3DF9FBDED|82e91796364c8f273e44700d2a3fff6f; ll=\"108288\"; ct=y; __utmz=30149280.1544249001.11.9.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; viewed=\"30299178_1040771_3406401_30383756_26630480_27028462_30320887_30152566_27094706_25985683\"; ap_v=0,6.0; __utma=30149280.26238068.1535123581.1544249001.1544261745.12; _pk_ref.100001.8cb4=%5B%22%22%2C%22%22%2C1544263181%2C%22https%3A%2F%2Fwww.baidu.com%2Flink%3Furl%3Dsq2OzKgUXY-Th9FWeXzZRNueRHfrN3yq00YD9IE8_BL6Plk-Zr0_bmxNVnhkkq4D%26wd%3D%26eqid%3D956622520005204f000000035c0b617a%22%5D; _pk_ses.100001.8cb4=*; ps=y; ue=\"wildpanda@yeah.net\"; _pk_id.100001.8cb4=61d736575660e131.1535123581.14.1544263609.1544260236."}]
def book_spider(book_tag):
    page_num=0
    book_list=[]
    try_times=0
    book_id = 0
    while(1):
        url='http://www.douban.com/tag/'+urllib2.quote(book_tag)+'/book?start='+str(page_num*15)
        print(url)
        time.sleep(np.random.rand())
        req = urllib2.Request(url, headers=hds[page_num%len(hds)])
        source_code = urllib2.urlopen(req).read()
        
        soup = BeautifulSoup(source_code,"html.parser")
        list_soup = soup.find('div', {'class': 'mod book-list'})
        
        try_times+=1
        if list_soup==None and try_times<100:
            continue
        elif list_soup==None or len(list_soup)<=1:
            break # Break when no informatoin got after 200 times requesting
        
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
                intro, aut_intro, short_review_url, long_review_url = get_intro(book_url)
                get_review(short_review_url,long_review_url,book_id)
            except:
                intro = ""
                aut_intro = ""
                people_num ='0'
            
            book_list.append([book_id, title, rating, author_info, pub_info, intro, aut_intro])
            book_list_buffer = np.array(book_list)
            book_list_buffer = book_list_buffer.reshape((-1,7))
            csvfile_book_list = pd.DataFrame(book_list_buffer,columns=["1","1","1","1","1","1","1"])
            #将dataframe写入文件
            csvfile_book_list_path = "bookname.csv"
            xlsfile_book_list_path = "bookname.xlsx"
            csvfile_book_list.to_csv(csvfile_book_list_path,encoding="utf-8",index=False)
            xlsfile_book_list = pd.read_csv(csvfile_book_list_path,encoding="utf-8")
            xlsfile_book_list.to_excel(xlsfile_book_list_path,sheet_name='data',index=False)
            print("Book info written!")
            book_id += 1
            try_times=0
        page_num+=1
        print ('Downloading Information From Page %d' % page_num)
    return book_list

def get_review(short_review_url,long_review_url,book_id):
    print("Begin to get comments and reviews!")
    time.sleep(np.random.rand())
    req = urllib2.Request(short_review_url, headers=hds[np.random.randint(0,len(hds))])
    source_code = urllib2.urlopen(req).read()
    soup = BeautifulSoup(source_code,"html.parser")
    comment_num = soup.find('span',{'id':'total-comments'}).get_text()
    comment_num = re.sub("\D", "", comment_num)
    comment_num = int(float(comment_num) / 20) + 1
    print("comments_num : %d"%comment_num)
    req = urllib2.Request(long_review_url, headers=hds[np.random.randint(0,len(hds))])
    source_code = urllib2.urlopen(req).read()
    soup = BeautifulSoup(source_code,"html.parser")
    review_num = soup.find('div',{'id':'content'}).h1.get_text()
    review_num = re.sub("\D", "", review_num)
    review_num = int(float(review_num) / 20) + 1
    print("reviews_num : %d"%review_num)
    
    comments = []
    reviews = []
    page_num = 0
    while(1):
        if(page_num != 0):
            comment_url = short_review_url + '/hot?p='+str(page_num)
        else:
            comment_url = short_review_url
        print(comment_url)
        print("process comment no %d, page %d"%(book_id,page_num + 1))
        if(page_num == comment_num - 1):
            break
        page_num += 1
        time.sleep(np.random.rand())
        req = urllib2.Request(comment_url, headers=hds[np.random.randint(0,len(hds))])
        source_code = urllib2.urlopen(req).read()
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
        if(page_num == comment_num):
            break
        print("process review no %d, page %d"%(book_id,page_num + 1))
        page_num += 1
        time.sleep(np.random.rand())
        req = urllib2.Request(review_url, headers=hds[np.random.randint(0,len(hds))])
        source_code = urllib2.urlopen(req).read()
        soup = BeautifulSoup(source_code,"html.parser")
        all_review = soup.find_all('div',{'class':'review-short'})
        for i in range(len(all_review)):
            full_review_id = all_review[i].get('data-rid')
            full_review_url = "https://book.douban.com/j/review/" + str(full_review_id) + "/full"
            print(full_review_url)
            time.sleep(np.random.rand())
            rev_req = urllib2.Request(full_review_url, headers=hds[np.random.randint(0,len(hds))])
            rev_source_code = urllib2.urlopen(rev_req).read()
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
    # comments_book_id = np.zeros_like(comments)
    # reviews_book_id = np.zeros_like(reviews)
    # comments = np.column_stack((comments_book_id, comments))
    # reviews = np.column_stack((reviews_book_id, reviews))
    # for i in range(len(comments)):
    #     comments[i][0] = str(book_id)
    # for i in range(len(reviews)):
    #     reviews[i][0] = str(book_id)

    csvfile_comments = pd.DataFrame(comments,columns=["短评论"])
    csvfile_reviews = pd.DataFrame(reviews,columns=["长评论"])
    #将dataframe写入文件
    csv_comments_path = "%d_short.csv"%(book_id)
    xls_comments_path = "%d_short.xlsx"%(book_id)
    csvfile_comments.to_csv(csv_comments_path,encoding="utf-8",index=False)
    xlsfile_comments = pd.read_csv(csv_comments_path,encoding="utf-8")
    xlsfile_comments.to_excel(xls_comments_path,sheet_name='data',index=False)

    csv_reviews_path = "%d_long.csv"%(book_id)
    xls_reviews_path = "%d_long.xlsx"%(book_id)
    csvfile_reviews.to_csv(csv_reviews_path,encoding="utf-8",index=False)
    xlsfile_reviews = pd.read_csv(csv_reviews_path,encoding="utf-8")
    xlsfile_reviews.to_excel(xls_reviews_path,sheet_name='data',index=False)

def get_intro(url):
    #try:
    opener = urllib2.build_opener()
    opener.addheaders.append(('Cookie', 'cookiename=cookievalue'))
    f = opener.open("http://example.com/")

    req = urllib2.Request(url, headers=hds[np.random.randint(0,len(hds))])
    source_code = urllib2.urlopen(req).read()
    # except (urllib2.HTTPError, urllib2.URLError), e:
    #     print e
    soup = BeautifulSoup(source_code,"html.parser")
    all_intro = soup.find_all('div',{'class':'intro'})
    short_review_url = url[:-13] + "comments"
    long_review_url = url[:-13] + "reviews"
    if(len(all_intro) > 3):
        intro = all_intro[1].get_text().strip()
        #print(all_intro[1].get_text().strip())
        aut_intro = all_intro[3].get_text().strip()
        #print(all_intro[3].get_text().strip())
    elif(len(all_intro) > 2):
        intro = all_intro[1].get_text().strip()
        #print(all_intro[1].get_text().strip())
        aut_intro = all_intro[2].get_text().strip()
        #print(all_intro[2].get_text().strip())
    else:
        intro = all_intro[0].get_text().strip()
        #print(all_intro[0].get_text().strip())
        aut_intro = all_intro[1].get_text().strip()
        #print(all_intro[1].get_text().strip())
    #people_num=soup.find('div',{'class':'rating_sum'}).findAll('span')[1].string.strip()
    return intro, aut_intro, short_review_url, long_review_url

def do_spider(book_tag_lists):
    book_lists=[]
    for book_tag in book_tag_lists:
        book_list=book_spider(book_tag)
        #book_list=sorted(book_list,key=lambda x:x[1],reverse=True)
        book_lists.append(book_list)
    return book_lists

if __name__=='__main__':
    #book_tag_lists = ['心理','判断与决策','算法','数据结构','经济','历史']
    #book_tag_lists = ['传记','哲学','编程','创业','理财','社会学','佛教']
    #book_tag_lists = ['思想','科技','科学','web','股票','爱情','两性']
    #book_tag_lists = ['计算机','机器学习','linux','android','数据库','互联网']
    #book_tag_lists = ['数学']
    #book_tag_lists = ['摄影','设计','音乐','旅行','教育','成长','情感','育儿','健康','养生']
    #book_tag_lists = ['商业','理财','管理']
    #book_tag_lists = ['名著']
    #book_tag_lists = ['科普','经典','生活','心灵','文学']
    #book_tag_lists = ['科幻','思维','金融']
    book_tag_lists = ['宗教']#['个人管理','时间管理','投资','文化','宗教']
    book_lists = do_spider(book_tag_lists)