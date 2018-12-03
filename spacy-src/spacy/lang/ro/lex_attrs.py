# coding: utf8
from __future__ import unicode_literals
from ...attrs import LIKE_NUM, IS_OOV
import os

_num_words = set("""
zero unu una doi doua trei patru cinci șase șapte opt nouă zece
unsprezece doisprezece treisprezece paisprezece cincisprezece șaisprezece șaptesprezece
optsprezece nouăsprezece douazeci treizeci patruzeci cincizeci șaizeci șaptezeci
optzeci nouăzeci sută mie milion miliard 
    """)

_ordinal_words = set("""
prima primul doua doilea treia treilea patra patrulea cincea cincilea 
șasea șaselea șaptelea optulea noua nouălea zecelea zecea unsprezecelea doisprezecelea
treisprezecelea paisprezecelea cincisprezecelea șaisprezecelea șaptesprezecea optsprezecea optsprezecelea
douăzeci treizeci cincizecilea șaizecilea șaptezecilea
""".split())


def like_num(text):
    # Might require more work?
    # See this discussion: https://github.com/explosion/spaCy/pull/1161
    text = text.replace(',', '').replace('.', '')
    if text.isdigit():
        return True
    if text.count('/') == 1:
        num, denom = text.split('/')
        if num.isdigit() and denom.isdigit():
            return True
    if text.lower() in _num_words:
        return True
    if text.lower() in _ordinal_words:
        return True
    return False

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
dict_file = "dict_ro.txt"
dict_path = os.path.join(script_dir, dict_file)
words = set()
with open(dict_path, "rt") as f:
    words = {line.strip() for line in f.readlines()}

def is_oov(text):
    return text not in words

LEX_ATTRS = {
    LIKE_NUM: like_num,
    IS_OOV: is_oov
}
