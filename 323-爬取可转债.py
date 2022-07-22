
# Python爬取可转债和正股涨跌幅信息

# 前几天有人在公众号留言，提到一个可转债的思路，想看下转债和正股的涨跌有没有关联，
# 如果转债的走势和正股一致，而且滞后于正股，那我们就可以筛选正股涨的多，而转债还没涨起来的来捡漏。
# 为了验证这个思路是否靠谱，昨晚花了两小时，写了个爬虫程序，粗略看了一下，好像转债和正股的涨跌关系不大，
# 这里把代码贴出来，感兴趣的可以深入研究，研究出什么门道记得告知我。
# 数据来源：东方财富（https://data.eastmoney.com/kzz/）

# 1、爬取列表页

import requests
import json
import re

def get_list():
    list_url = 'https://datacenter-web.eastmoney.com/api/data/v1/get?sortColumns=PUBLIC_START_DATE&sortTypes=-1&pageSize=50&pageNumber={num}&reportName=RPT_BOND_CB_LIST&columns=ALL&quoteColumns=f2~01~CONVERT_STOCK_CODE~CONVERT_STOCK_PRICE%2Cf235~10~SECURITY_CODE~TRANSFER_PRICE%2Cf236~10~SECURITY_CODE~TRANSFER_VALUE%2Cf2~10~SECURITY_CODE~CURRENT_BOND_PRICE%2Cf237~10~SECURITY_CODE~TRANSFER_PREMIUM_RATIO%2Cf239~10~SECURITY_CODE~RESALE_TRIG_PRICE%2Cf240~10~SECURITY_CODE~REDEEM_TRIG_PRICE%2Cf23~01~CONVERT_STOCK_CODE~PBV_RATIO&source=WEB&client=WEB'
    i = 6
    while True:
        response = requests.get(list_url.format(num=i))
        response_dict = json.loads(response.text)
        if not response_dict['success']:
            break
        data = response_dict['result']['data']
        parse_list(data)
        print('>> page:', i, 'ok')
        i += 1


def to_txt(line):
    with open('./secu.csv', 'a') as f:
        f.write(line)

if __name__ == '__main__':
    get_list()


# 2、解析列表数据

def parse_list(data):
    for row in data:
        try:
            secu_name = row['SECURITY_NAME_ABBR']
            secu_code = row['SECUCODE']
            stock_code = row['CONVERT_STOCK_CODE']
            stock_name = row['SECURITY_SHORT_NAME']
            secu_detail = get_secu_detail(secu_code)
            if not secu_detail:
                continue
            stock_detail = get_stock_detail(stock_code)
            to_txt(','.join([secu_name, secu_code[:6], *list(map(str, secu_detail)), \
                stock_name, stock_code, *list(map(str, stock_detail)), \
                str(round(stock_detail[2] - secu_detail[2],3))]) + '\n')
        except:
            print(secu_code)


# 3、爬取转债价格

def get_secu_detail(code):
    # 获取带市场前缀的code
    code = (code[-2:] + code[:6]).lower()
    url = f'https://quote.eastmoney.com/bond/{code}.html'
    html = requests.get(url).text
    rex = re.search('quotecode":"(.*?)"', html)
    m_code = rex.group(1)
    # 获取接口内容
    api_url = 'https://push2.eastmoney.com/api/qt/stock/details/get?pos=-12&secid={code}&fields1=f1,f2,f3,f4&fields2=f51,f52,f53,f54,f55&fltt=2'
    response = requests.get(api_url.format(code=m_code))
    response_dict = json.loads(response.text)
    if (not response_dict) or (not response_dict['data']):
        return None
    pre_price = float(response_dict['data']['prePrice'])
    details = response_dict['data']['details']
    if len(details):
        cur_price = float(details[-1].split(',')[1])
        pct_chg = round((cur_price - pre_price) * 100 / pre_price, 3)
        return pre_price, cur_price, pct_chg

# 4、解析股票价格

def get_stock_detail(code):
    # 获取带市场前缀的code
    url = f'https://data.eastmoney.com/stockdata/{code}.html'
    html = requests.get(url).text
    rex = re.search('hqCode":"(.*?)"', html)
    m_code = rex.group(1)
    # 获取接口内容
    api_url = 'https://push2.eastmoney.com/api/qt/stock/get?fltt=2&invt=2&secid={code}&fields=f57,f58,f107,f43,f169,f170,f171,f47,f48,f60,f46,f44,f45,f168,f50,f162,f177'
    response = requests.get(api_url.format(code=m_code))
    response_dict = json.loads(response.text)
    if (not response_dict) or (not response_dict['data']):
        return None
    pre_price = float(response_dict['data']['f60'])
    cur_price = float(response_dict['data']['f43'])
    pct_chg = round((cur_price - pre_price) * 100 / pre_price, 3)
    return pre_price, cur_price, pct_chg




