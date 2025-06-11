#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#该列表声明了当你使用 from model import * 时，哪些模块或对象是可以导出的。
#这里只导出了 'BGCN' ，意味着仅有该模块（或对象）是可见的。
#但注意下面几行注释的导入，如果取消注释，则可以导入更多模块。
__all__ = ['BGCN']

# from .BGCN import BGCN, BGCN_Info
# from .SMGCN import *
# from .SMGCN_pre import *
