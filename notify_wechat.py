#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微信通知脚本 - 支持 虾推啥(xtuis) 和企业微信两种方式
"""
import requests
import argparse
import json
from datetime import datetime
import os

class WeChatNotifier:
    """微信通知类"""
    
    def __init__(self, method='serverchan', **kwargs):
        """
        初始化通知器
        
        Args:
            method: 'serverchan' 或 'qywx' (企业微信)
            **kwargs: 根据 method 传入不同的参数
                - serverchan: sendkey (这里作为虾推啥 token 使用)
                - qywx: corpid, corpsecret, agentid
        """
        self.method = method
        if method == 'serverchan':
            # 这里将 sendkey 视为虾推啥(xtuis) 的 token
            self.sendkey = kwargs.get('sendkey')
            if not self.sendkey:
                raise ValueError("虾推啥方式需要提供 token (sendkey)")
        elif method == 'qywx':
            self.corpid = kwargs.get('corpid')
            self.corpsecret = kwargs.get('corpsecret')
            self.agentid = kwargs.get('agentid')
            if not all([self.corpid, self.corpsecret, self.agentid]):
                raise ValueError("企业微信方式需要提供 corpid, corpsecret, agentid")
            self.access_token = None
        else:
            raise ValueError(f"不支持的通知方式: {method}")
    
    def _get_qywx_token(self):
        """获取企业微信 access_token"""
        url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken"
        params = {
            'corpid': self.corpid,
            'corpsecret': self.corpsecret
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get('errcode') == 0:
                self.access_token = data.get('access_token')
                return True
            else:
                print(f"获取企业微信 token 失败: {data.get('errmsg')}")
                return False
        except Exception as e:
            print(f"获取企业微信 token 异常: {e}")
            return False
    
    def send_serverchan(self, title, content):
        """通过 虾推啥(xtuis) 发送消息（兼容 serverchan 方法名）"""
        # 按虾推啥文档，使用 text / desp 字段
        url = f"https://wx.xtuis.cn/{self.sendkey}.send"
        data = {
            "text": title,
            "desp": content,  # 最大支持 64KB 文本
        }
        try:
            resp = requests.post(url, data=data, timeout=10)
            resp.raise_for_status()
            # 虾推啥通常返回 JSON 或纯文本；这里只要 HTTP 200 就认为成功
            return True, "发送成功"
        except Exception as e:
            return False, f"请求异常: {e}"
    
    def send_qywx(self, title, content):
        """通过企业微信发送消息"""
        if not self.access_token:
            if not self._get_qywx_token():
                return False, "获取 access_token 失败"
        
        url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send"
        params = {'access_token': self.access_token}
        
        # 构建消息体
        message = {
            "touser": "@all",  # 发送给所有人，也可以指定用户
            "msgtype": "text",
            "agentid": self.agentid,
            "text": {
                "content": f"{title}\n\n{content}"
            }
        }
        
        try:
            resp = requests.post(url, params=params, json=message, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            if result.get('errcode') == 0:
                return True, "发送成功"
            else:
                # token 可能过期，重试一次
                if result.get('errcode') == 40014:
                    self.access_token = None
                    return self.send_qywx(title, content)
                return False, f"发送失败: {result.get('errmsg', '未知错误')}"
        except Exception as e:
            return False, f"请求异常: {e}"
    
    def send(self, title, content):
        """发送通知"""
        if self.method == 'serverchan':
            return self.send_serverchan(title, content)
        elif self.method == 'qywx':
            return self.send_qywx(title, content)


def main():
    parser = argparse.ArgumentParser(description='微信通知脚本')
    parser.add_argument('--title', required=True, help='通知标题')
    parser.add_argument('--body', required=True, help='通知内容')
    parser.add_argument('--method', choices=['serverchan', 'qywx'], default='serverchan',
                       help='通知方式: serverchan (虾推啥/xtuis) 或 qywx (企业微信)')
    
    # 虾推啥(xtuis) 参数
    parser.add_argument('--sendkey', help='虾推啥 token (或设置环境变量 SENDKEY)')
    
    # 企业微信参数
    parser.add_argument('--corpid', help='企业微信 CorpID (或设置环境变量 QYWX_CORPID)')
    parser.add_argument('--corpsecret', help='企业微信 CorpSecret (或设置环境变量 QYWX_CORPSECRET)')
    parser.add_argument('--agentid', help='企业微信 AgentID (或设置环境变量 QYWX_AGENTID)')
    
    args = parser.parse_args()
    
    # 从环境变量读取配置（优先级：命令行参数 > 环境变量）
    if args.method == 'serverchan':
        sendkey = args.sendkey or os.getenv('SENDKEY')
        if not sendkey:
            print("错误: 虾推啥方式需要提供 --sendkey 或设置环境变量 SENDKEY")
            return 1
        notifier = WeChatNotifier(method='serverchan', sendkey=sendkey)
    else:
        corpid = args.corpid or os.getenv('QYWX_CORPID')
        corpsecret = args.corpsecret or os.getenv('QYWX_CORPSECRET')
        agentid = args.agentid or os.getenv('QYWX_AGENTID')
        if not all([corpid, corpsecret, agentid]):
            print("错误: 企业微信方式需要提供 corpid, corpsecret, agentid")
            return 1
        notifier = WeChatNotifier(method='qywx', corpid=corpid, corpsecret=corpsecret, agentid=agentid)
    
    # 发送通知
    success, msg = notifier.send(args.title, args.body)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if success:
        print(f"[{timestamp}] ✅ 微信通知发送成功: {msg}")
        return 0
    else:
        print(f"[{timestamp}] ❌ 微信通知发送失败: {msg}")
        return 1


if __name__ == '__main__':
    exit(main())
