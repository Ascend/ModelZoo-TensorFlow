# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encoders for text data.

* TextEncoder: base class
* ByteTextEncoder: for ascii text
* TokenTextEncoder: with user-supplied vocabulary file
* SubwordTextEncoder: invertible
"""

import datetime
import re

import six
import tensorflow.compat.v1 as tf
from six.moves import range  # pylint: disable=redefined-builtin

from network.tensor2tensor import tokenizer

# Reserved tokens for things like padding and EOS symbols.
PAD = "<pad>"
EOS = "<EOS>"
RESERVED_TOKENS = [PAD, EOS]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)  # Normally 0
EOS_ID = RESERVED_TOKENS.index(EOS)  # Normally 1
_UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")
_ESCAPE_CHARS = set(u"\\_u;0123456789")


def _strip_ids(ids, ids_to_strip):
    """Strip ids_to_strip from the end ids."""
    ids = list(ids)
    while ids and ids[-1] in ids_to_strip:
        ids.pop()
    return ids


def _is_unicode(s):
    return isinstance(s, six.text_type)


def native_to_unicode(s):
    if _is_unicode(s):
        return s
    try:
        return _to_unicode(s)
    except UnicodeDecodeError:
        res = _to_unicode(s, ignore_errors=True)
        print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                        "E", "Ignoring Unicode error, outputting: %s" % res))
        return res


def _to_unicode(s, ignore_errors=False):
    if _is_unicode(s):
        return s
    error_mode = "ignore" if ignore_errors else "strict"
    return s.decode("utf-8", errors=error_mode)


def _escape_token(token, alphabet):
    """Escape away underscores and OOV characters and append '_'.

    This allows the token to be expressed as the concatenation of a list
    of subtokens from the vocabulary. The underscore acts as a sentinel
    which allows us to invertibly concatenate multiple such lists.

    Args:
        token: A unicode string to be escaped.
        alphabet: A set of all characters in the vocabulary's alphabet.

    Returns:
        escaped_token: An escaped unicode string.

    Raises:
        ValueError: If the provided token is not unicode.
    """
    if not isinstance(token, six.text_type):
        raise ValueError("Expected string type for token, got %s" % type(token))

    token = token.replace(u"\\", u"\\\\").replace(u"_", u"\\u")
    ret = [c if c in alphabet and c != u"\n" else r"\%d;" % ord(c) for c in token]
    return u"".join(ret) + "_"


class TextEncoder(object):
    """Base class for converting from ints to/from human readable strings."""

    def __init__(self, num_reserved_ids=NUM_RESERVED_TOKENS):
        self._num_reserved_ids = num_reserved_ids

    @property
    def num_reserved_ids(self):
        return self._num_reserved_ids

    def encode(self, s):
        """Transform a human-readable string into a sequence of int ids.

        The ids should be in the range [num_reserved_ids, vocab_size). Ids [0,
        num_reserved_ids) are reserved.

        EOS is not appended.

        Args:
            s: human-readable string to be converted.

        Returns:
            ids: list of integers
        """
        return [int(w) + self._num_reserved_ids for w in s.split()]

    def decode(self, ids, strip_extraneous=False):
        """Transform a sequence of int ids into a human-readable string.

        EOS is not expected in ids.

        Args:
            ids: list of integers to be converted.
            strip_extraneous: bool, whether to strip off extraneous tokens
                (EOS and PAD).

        Returns:
            s: human-readable string.
        """
        if strip_extraneous:
            ids = _strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        return " ".join(self.decode_list(ids))

    def decode_list(self, ids):
        """Transform a sequence of int ids into a their string versions.

        This method supports transforming individual input/output ids to their
        string versions so that sequence to/from text conversions can be visualized
        in a human readable format.

        Args:
            ids: list of integers to be converted.

        Returns:
            strs: list of human-readable string.
        """
        decoded_ids = []
        for id_ in ids:
            if 0 <= id_ < self._num_reserved_ids:
                decoded_ids.append(RESERVED_TOKENS[int(id_)])
            else:
                decoded_ids.append(id_ - self._num_reserved_ids)
        return [str(d) for d in decoded_ids]

    @property
    def vocab_size(self):
        raise NotImplementedError()


class SubwordTextEncoder(TextEncoder):
    """Class for invertibly encoding text using a limited vocabulary.

    Invertibly encodes a native string as a sequence of subtokens from a limited
    vocabulary.

    A SubwordTextEncoder is built from a corpus (so it is tailored to the text in
    the corpus), and stored to a file. See text_encoder_build_subword.py.

    It can then be loaded and used to encode/decode any text.

    Encoding has four phases:

    1. Tokenize into a list of tokens.  Each token is a unicode string of either
        all alphanumeric characters or all non-alphanumeric characters.  We drop
        tokens consisting of a single space that are between two alphanumeric
        tokens.

    2. Escape each token.  This escapes away special and out-of-vocabulary
        characters, and makes sure that each token ends with an underscore, and
        has no other underscores.

    3. Represent each escaped token as a the concatenation of a list of subtokens
        from the limited vocabulary.  Subtoken selection is done greedily from
        beginning to end.  That is, we construct the list in order, always picking
        the longest subtoken in our vocabulary that matches a prefix of the
        remaining portion of the encoded token.

    4. Concatenate these lists.  This concatenation is invertible due to the
        fact that the trailing underscores indicate when one list is finished.

    """

    def __init__(self, filename=None):
        """Initialize and read from a file, if provided.

        Args:
            filename: filename from which to read vocab. If None, do not load a
                vocab
        """
        self._alphabet = set()
        self.filename = filename
        if filename is not None:
            self._load_from_file(filename)
        super(SubwordTextEncoder, self).__init__()

    def encode(self, s):
        """Converts a native string to a list of subtoken ids.

        Args:
            s: a native string.
        Returns:
            a list of integers in the range [0, vocab_size)
        """
        return self._tokens_to_subtoken_ids(tokenizer.encode(native_to_unicode(s)))

    def _tokens_to_subtoken_ids(self, tokens):
        """Converts a list of tokens to a list of subtoken ids.

        Args:
            tokens: a list of strings.
        Returns:
            a list of integers in the range [0, vocab_size)
        """
        ret = []
        for token in tokens:
            ret.extend(self._token_to_subtoken_ids(token))
        return ret

    def _token_to_subtoken_ids(self, token):
        """Converts token to a list of subtoken ids.

        Args:
            token: a string.
        Returns:
            a list of integers in the range [0, vocab_size)
        """
        cache_location = hash(token) % self._cache_size
        cache_key, cache_value = self._cache[cache_location]
        if cache_key == token:
            return cache_value
        ret = self._escaped_token_to_subtoken_ids(_escape_token(token, self._alphabet))
        self._cache[cache_location] = (token, ret)
        return ret

    def _escaped_token_to_subtoken_ids(self, escaped_token):
        """Converts an escaped token string to a list of subtoken IDs.

        Args:
            escaped_token: An escaped token as a unicode string.
        Returns:
            A list of subtoken IDs as integers.
        """
        return [
            self._subtoken_string_to_id[subtoken]
            for subtoken in self._escaped_token_to_subtoken_strings(escaped_token)
        ]

    def _escaped_token_to_subtoken_strings(self, escaped_token):
        """Converts an escaped token string to a list of subtoken strings.

        Args:
            escaped_token: An escaped token as a unicode string.
        Returns:
            A list of subtokens as unicode strings.
        """
        # NOTE: This algorithm is greedy; it won't necessarily produce the "best"
        # list of subtokens.
        ret = []
        start = 0
        token_len = len(escaped_token)
        while start < token_len:
            for end in range(
                    min(token_len, start + self._max_subtoken_len), start, -1):
                subtoken = escaped_token[start:end]
                if subtoken in self._subtoken_string_to_id:
                    ret.append(subtoken)
                    start = end
                    break

            else:  # Did not break
                # If there is no possible encoding of the escaped token then one of the
                # characters in the token is not in the alphabet. This should be
                # impossible and would be indicative of a bug.
                assert False, "Token substring not found in subtoken vocabulary."

        return ret

    def _init_subtokens_from_list(self, subtoken_strings, reserved_tokens=None):
        """Initialize token information from a list of subtoken strings.

        Args:
            subtoken_strings: a list of subtokens
            reserved_tokens: List of reserved tokens. We must have `reserved_tokens`
                as None or the empty list, or else the global variable `RESERVED_TOKENS`
                must be a prefix of `reserved_tokens`.

        Raises:
            ValueError: if reserved is not 0 or len(RESERVED_TOKENS). In this case, it
                is not clear what the space is being reserved for, or when it will be
                filled in.
        """
        if reserved_tokens is None:
            reserved_tokens = []

        if reserved_tokens:
            self._all_subtoken_strings = reserved_tokens + subtoken_strings
        else:
            self._all_subtoken_strings = subtoken_strings

        # we remember the maximum length of any subtoken to avoid having to
        # check arbitrarily long strings.
        self._max_subtoken_len = max([len(s) for s in subtoken_strings])
        self._subtoken_string_to_id = {
            s: i + len(reserved_tokens)
            for i, s in enumerate(subtoken_strings) if s
        }
        # Initialize the cache to empty.
        self._cache_size = 2 ** 20
        self._cache = [(None, None)] * self._cache_size

    def _init_alphabet_from_tokens(self, tokens):
        """Initialize alphabet from an iterable of token or subtoken strings."""
        # Include all characters from all tokens in the alphabet to guarantee that
        # any token can be encoded. Additionally, include all escaping characters.
        self._alphabet = {c for token in tokens for c in token}
        self._alphabet |= _ESCAPE_CHARS

    def _load_from_file_object(self, f):
        """Load from a file object.

        Args:
            f: File object to load vocabulary from
        """
        subtoken_strings = []
        for line in f:
            s = line.rstrip()
            # Some vocab files wrap words in single quotes, but others don't
            if ((s.startswith("'") and s.endswith("'")) or
                    (s.startswith("\"") and s.endswith("\""))):
                s = s[1:-1]
            subtoken_strings.append(native_to_unicode(s))
        self._init_subtokens_from_list(subtoken_strings)
        self._init_alphabet_from_tokens(subtoken_strings)

    def _load_from_file(self, filename):
        """Load from a vocab file."""
        if not tf.gfile.Exists(filename):
            raise ValueError("File %s not found" % filename)
        with tf.gfile.Open(filename) as f:
            self._load_from_file_object(f)
