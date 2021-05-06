# -*- coding:utf8 -*-
"""
@Author: Zhirui(Alex) Yang
@Date: 2021/5/1 下午11:26
"""

import os


__project_dir = os.path.dirname(__file__)
DATA_DIR = os.path.join(__project_dir, 'data')


if __name__ == '__main__':
    print(DATA_DIR)