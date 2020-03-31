from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from urllib import request
import os

def waitSession(val, tt):
    try:
        element = WebDriverWait(driver, tt).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, val))
        )
    except TimeoutException:
        print("타임아웃")
        
        raise Exception('타임아웃')


st = int(input("시작페이지 입력 >"))
ERROR_TRY = int(input("에러 재시도 횟수 입력 >"))

if st == -1 :
    f = open("pagecount.txt", "r")
    st = int(f.readline())
    f.close()

print(str(st) + "번째 페이지부터 로드를 시작합니다.")

options = webdriver.ChromeOptions()
driver = webdriver.Chrome("C:\chromedriver\chromedriver.exe", options=options)

folder = "images/"
if not os.path.exists(folder):
    os.makedirs(folder)
    


# 1~ 85 페이지 url로 접근
for i in range(st, 10):
    print(str(i)+"번 페이지 접근 시작")
    for m in range(1, ERROR_TRY+1):
        try:
            #page_url = "http://avangs.info/index.php?mid=resource_200x&category=1126403&listStyle=list&page=" + str(i)
            page_url = "http://avangs.info/index.php?mid=derivative_works&category=1405910&page=" + str(i)
            driver.get(page_url)
            waitSession("#bd_237788_0 > div.bd_lst_wrp > table > tbody", 100)
            board = driver.find_element_by_css_selector("#bd_237788_0 > div.bd_lst_wrp > table > tbody")
            posts = board.find_elements_by_css_selector("tr")
            url_list = []
            break
        except:
            print("URL 접슨 에러 정지 " + page_url)
            if m==ERROR_TRY :
                print("재시도 종료")
                break
            print("재시도 "  + str(m) + "번째")
    # 해당 페이지에 게시판에 공지 제외 url를 긁어옴
    for post in posts:
        if "notice" in post.get_attribute("class"):
            continue
        url_list.append(post.find_element_by_css_selector("td.title > a").get_attribute("href"))
    #print(url_list)
    
    # 얻은 url을 기반으로 첨부파일 가져옴
    for url in url_list:
        for j in range(1, ERROR_TRY+1):
            try:
                driver.get(url)
                waitSession("#content  table.bd_tb", 100)
                contents = driver.find_element_by_css_selector("#content  table.bd_tb")
                contents = contents.find_elements_by_css_selector("tbody > tr > td > ul > li > a")
                for content in contents:
                    for k in range(1, ERROR_TRY+1):
                        try : 
                            savename = folder+content.text
                            download_url = content.get_attribute("href")
                            request.urlretrieve(download_url, savename)
                            break
                        except :
                            print("첨부파일 로드 오류 : {0}".format(str(content)))
                            if k==ERROR_TRY :
                                print("재시도 종료")
                                break
                            print("재시도 "  + str(k) + "번째")
                            
                break
            except:
                print(" 첨부파일 접근 에러 정지 " + url)
                if j==ERROR_TRY :
                    print("재시도 종료")
                    break
                print("재시도 "  + str(j) + "번째")
                
                #exit()
    print(str(i)+"번 페이지 접근 완료")
    f = open("pagecount.txt", "w")
    f.write(str(i))
    f.close()
    print(str(i)+"번 페이지 카운트 완료")
