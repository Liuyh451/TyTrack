import http.client
import hashlib
import base64
from datetime import datetime

import requests


def generate_access_key(user_id, security_key):
    # 获取北京时间，格式为yyyyMMddHHmmss
    beijing_time = datetime.now().strftime('%Y%m%d%H%M%S')
    print(beijing_time)

    # Step 1: 拼接用户ID和时间
    value = user_id + beijing_time
    print(value)
    # Step 2: 拼接value和securityKey后进行MD5加密
    md5_string = hashlib.md5((value + security_key).encode('utf-8')).hexdigest()

    # Step 3: 拼接value和MD5值，并进行Base64编码
    signature = value + md5_string
    typhoon_access_key = base64.b64encode(signature.encode('utf-8')).decode('utf-8')

    return typhoon_access_key



