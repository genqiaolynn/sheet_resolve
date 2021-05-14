# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/2/6 16:14'


'''
except Exception,e: 中的e只用来获取异常值，print e打印出异常值，这样不会打断程序运行； 
而traceback.print_exc()语句与不进行异常处理的运行效果是一致的，都会打印出异常类型，异常值，出错位置，并且打断程序运行。 
so，如果想要获取异常完整信息而不打断程序，最好使用sys.exc_info(),简单的用法是 except Exception: print(sys.exc_info()[0:2]) 
# 打印错误类型，错误值 print(traceback.extract_tb(sys.exc_info()[2])) #出错位置
'''


import traceback, sys
try:
    1/0
except Exception as e:
    # print(e)    # 打印出异常值
    traceback.print_exc()    # 打印出异常信息，并打断程序
    # print(sys.exc_info()[:2])  # 获取异常完整信息并且不打断程序
    # print(traceback.extract_tb(sys.exc_info()[2]))  # 出错位置 [<FrameSummary file F:/exam/sheet_resolve/demo11.py, line 19 in <module>>]