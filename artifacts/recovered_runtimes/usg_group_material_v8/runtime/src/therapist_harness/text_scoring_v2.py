"""Opt-in orthographic correction; frozen legacy CER remains reproducible.

銳/鋭: https://dict.variants.moe.edu.tw/dictView.jsp?ID=47272&la=0
拐杖/枴杖: https://words.hk/zidin/%E6%8B%90%E6%9D%96
Legacy OpenCC s2t also renders 拐杖 as 柺杖.
"""
from therapist_harness.text import text_normalize

_VARIANTS=str.maketrans({'鋭':'銳','枴':'柺','拐':'柺'})


def text_normalize_v2(text: str) -> str:
    return text_normalize(text).translate(_VARIANTS)
