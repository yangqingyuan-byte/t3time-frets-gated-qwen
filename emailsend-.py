# from playwright.async_api import async_playwright
# import asyncio
# import regex
# import json
# import logging
# import os
# import random
import smtplib
from datetime import datetime
# from urllib.parse import urlparse, parse_qs
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# import aiohttp
# import ssl


def send_email(un, pd, periodName, deviceName, receiver):
    sender = '1302905387@qq.com'
    subject = f'{periodName}#{deviceName}自动{datetime.now().strftime("%H:%M:%S.%f")}'
    message = f'账号：{un}, 密码：{pd}, {periodName}{deviceName}。请登录智慧东大付款，如果要出场地的话账号密码不要透露给别人，自己来扫码，不然球馆会打电话催'

    try:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = receiver
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))

        with smtplib.SMTP('smtp.qq.com', 587) as server:
            server.starttls()
            server.login(sender, 'vyfjlrupoebojfdh')
            server.sendmail(sender, receiver, msg.as_string())
    except smtplib.SMTPResponseException as e:
        if e.smtp_code == -1 and e.smtp_error == b'\x00\x00\x00':
            # 忽略特定异常
            print(e)
            pass
        else:
            error_msg = f'邮件发送失败 (SMTP 错误): {e},当前时间为 {datetime.now().strftime("%H:%M:%S.%f")}'
            print(error_msg)
    except Exception as e:
        error_msg = f'邮件发送失败 (其他错误): {e},当前时间为 {datetime.now().strftime("%H:%M:%S.%f")}'
        print(error_msg)
        
        
        
un='1'
pd='1'
periodName='1'
deviceName='1'
receiver='1124998618@qq.com'
send_email(un, pd, periodName, deviceName, receiver)