# coding=utf-8
# =============================================
# @Time      : 2023-02-15 16:14
# @Author    : DongWei1998
# @FileName  : flasktest.py
# @Software  : PyCharm
# =============================================


import json,time,requests

def translator():
    url = f"http://127.0.0.1:5557/api/v1/translator"
    demo_text ={
        'text':"este é o primeiro livro que eu fiz."
    }

    headers = {
        'Content-Type': 'application/json'
    }
    start = time.time()
    result = requests.post(url=url, json=demo_text,headers=headers)
    end = time.time()
    if result.status_code == 200:
        obj = json.loads(result.text)
        print(obj)
    else:
        print(result)
    print('Running time: %s Seconds' % (end - start))


translator()