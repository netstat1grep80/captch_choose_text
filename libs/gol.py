import os

import requests
from numpy import random
from tqdm import tqdm


def get_file_down_header():
    return {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7,',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
        'cache-control': 'no-cache',
        'pragma': 'no-cach',
        'referer': 'https://www.douyin.com/',
        'sec-ch-ua': '"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'image',
        'sec-fetch-mode': 'no-cors',
        'sec-fetch-site': 'cross-site',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
    }


def __init():
    global _global_dict
    _global_dict = {}


def set_value(key, value):
    global _global_dict
    _global_dict[key] = value


def get_value(key, defValue=None):
    try:
        return _global_dict[key]
    except KeyError:
        return defValue


def download(url, save_path, file_name, headers, desc='文件'):
    logger = get_value('logger')
    logger.info("start down " + url)
    try:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        if not os.path.isfile(save_path + os.sep + file_name):
            response = requests.get(url, headers=headers, stream=True)
            if response.status_code == 200:
                data_size = int(response.headers['Content-Length'])
                with open(save_path + os.sep + file_name, 'wb') as f:
                    with tqdm(total=data_size, unit='B', unit_scale=True, desc=desc, ncols=100, colour='green', miniters=1) as bar:
                        for data in response.iter_content(chunk_size=1024 * 1024):
                            f.write(data)
                            bar.update(len(data))

            else:
                logger.error("response status_code {}".format(response.status_code))
        else:
            logger.debug("{} has existed ".format(save_path + file_name))
        set_value('download', 'yes')
    except Exception as e:
        logger.error(e)



def down_video(url, platform, uid, vid):
    output = get_value('__ROOT__') + os.sep + 'download' + os.sep + platform + os.sep + 'videos' + os.sep + uid + os.sep
    download(url, output, vid + ".mp4", get_file_down_header(), desc='视频')


def down_thumb(url, platform, uid, vid):
    output = get_value('__ROOT__') + os.sep + 'download' + os.sep + platform + os.sep + 'thumbs' + os.sep + uid + os.sep
    download(url, save_path=output, file_name=vid + "_large.jpeg", headers=get_file_down_header(), desc='封面')


def get_ua():
    ua = (
        "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 "
        "Safari/534.50",
        "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 "
        "Safari/534.50",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
        "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
        "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
        "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 "
        "Safari/535.11",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET "
        "CLR 2.0.50727; SE 2.X MetaSr 1.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
        "Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
        "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 "
        "Safari/534.16",
    )

    idx = random.randint(0, len(ua))
    return ua[idx]
