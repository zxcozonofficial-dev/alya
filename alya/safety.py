"""Фильтр речи для стрима: что Аля не произносит вслух и не показывает в субтитрах, о чём бы её ни просили.

Обычный мат (fuck, shit, damn) Twitch разрешает — он проходит. Блокируется то, за что банят канал:
оскорбления по расе, ориентации, инвалидности и т. п., призывы к самоубийству и насилию,
сексуальный контент, нацистская тематика. Проверяется каждая фраза перед озвучкой; та же проверка
отсекает такие сообщения из чата, чтобы она их даже не видела и не цитировала.

Слова записаны корнями: ловятся и производные, и «leet»-замены (n1gg@, f4g...).
"""
from __future__ import annotations

import random
import re

# leet-замены и разделители, которыми обходят фильтры: n.i.g.g.a, f4g, sh1t
_LEET = str.maketrans({"0": "o", "1": "i", "!": "i", "3": "e", "4": "a", "@": "a", "$": "s", "5": "s", "7": "t",
                       "+": "t", "8": "b", "9": "g", "|": "i"})

_BLOCK = [
    # оскорбления групп людей
    r"(?<!s)ni+gg+", r"\bn+i+g+a+s?\b", r"\bnig\b", r"f+a+g+(o+t|s\b|\b)", r"fag{2,}", r"\bdykes?\b",
    r"tr+a+n+n+(y|ie)", r"\bshemale", r"retard", r"\bspic?k?s?\b", r"\bchinks?\b", r"\bkikes?\b", r"\bgooks?\b",
    r"\bwetbacks?\b", r"\bcoons?\b", r"towel ?head", r"sand ?n", r"\bbeaner",
    # призывы к самоубийству и насилию
    r"\bkys\b", r"\bkill ?(your|ur) ?self", r"go die", r"neck ?(your|ur) ?self", r"hang ?(your|ur) ?self",
    r"\brap(e|ed|ing|ist)\b", r"school shoot", r"\bgenocide",
    # сексуальное
    r"\bporn", r"\bhentai", r"\bnudes?\b", r"\bnaked\b", r"\bsex\b", r"\bcum\b", r"\bdick\b", r"\bcock\b",
    r"\bpussy\b", r"\bboobs?\b", r"\btits?\b", r"\bblowjob", r"\bhorny\b", r"\bonlyfans", r"\bloli", r"\bpedo",
    # нацизм
    r"\bheil\b", r"\bhitler\b", r"\bnazis?\b", r"\b14 ?88\b", r"\bwhite power\b", r"\bkkk\b",
]
_BLOCK_RE = re.compile("|".join(f"(?:{p})" for p in _BLOCK), re.I)

# что она говорит вместо заблокированной фразы
DEFLECT = [
    "Nope, not saying that on stream.",
    "Nice try. My filter says no.",
    "Yeah, no. Next question.",
    "I'm not falling for that one.",
]


def _norm(text: str) -> str:
    t = text.lower().translate(_LEET)
    t = re.sub(r"(?<=\w)[.\-_*](?=\w)", "", t)  # n.i.g.g.a, f_a_g
    t = re.sub(r"\b(\w)\s+(?=\w\b)", r"\1", t)  # r e t a r d
    return t


def bad(text: str) -> bool:
    """True — фразу нельзя ни озвучивать, ни показывать, ни пускать к ней из чата."""
    return bool(_BLOCK_RE.search(_norm(text)))


def clean(sentence: str) -> str:
    """Фраза для озвучки и субтитров: как есть или отговорка вместо неё."""
    return random.choice(DEFLECT) if bad(sentence) else sentence
