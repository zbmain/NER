#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# find ./ -name "*" -type f -size 0c | xargs -n 1 rm -f
def delEmptyFile(file):
    print(os.path.getsize(file))
    os.path.getsize(file) or os.remove(file)  # 文件大小为0：删掉


def delDirEmptyFile(path):
    files = os.listdir(path)
    for file in files:
        os.path.isfile(file) and delEmptyFile(file)


if __name__ == "__main__":
    delDirEmptyFile('./')
