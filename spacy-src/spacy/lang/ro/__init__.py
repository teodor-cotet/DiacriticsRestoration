# coding: utf8
from __future__ import unicode_literals

from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
from .stop_words import STOP_WORDS
from .lemmatizer import LOOKUP
from .lex_attrs import LEX_ATTRS
from .punctuation import TOKENIZER_SUFFIXES, TOKENIZER_INFIXES
from .syntax_iterators import SYNTAX_ITERATORS

from ..tokenizer_exceptions import BASE_EXCEPTIONS
from ..norm_exceptions import BASE_NORMS
from ...language import Language
from ...attrs import LANG, NORM
from ...util import update_exc, add_lookups


class RomanianDefaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    lex_attr_getters.update(LEX_ATTRS)
    lex_attr_getters[LANG] = lambda text: 'ro'
    lex_attr_getters[NORM] = add_lookups(Language.Defaults.lex_attr_getters[NORM], BASE_NORMS)
    tokenizer_exceptions = update_exc(BASE_EXCEPTIONS, TOKENIZER_EXCEPTIONS)
    stop_words = STOP_WORDS
    infixes = TOKENIZER_INFIXES
    suffixes = TOKENIZER_SUFFIXES
    syntax_iterators = SYNTAX_ITERATORS
    lemma_lookup = LOOKUP

class Romanian(Language):
    lang = 'ro'
    Defaults = RomanianDefaults


__all__ = ['Romanian']

