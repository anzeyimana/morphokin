import math
import os
import re
from typing import Dict
from typing import Union, Tuple, Optional, List

import numpy as np
from minineedle import needle, core

from morphokin.data.language_data import all_wt_abbrevs, all_word_types, all_pos_tags, all_wt_abbrevs_dict
from morphokin.data.syllable_vocab import KINSPEAK_VOCAB_IDX, text_to_id_sequence, id_sequence_to_text
from morphokin.data.uds_client import UnixSocketClient

PAD_ID = 0
UNK_ID = 1
MSK_ID = 2
BOS_ID = 3
EOS_ID = 4

english_BOS_idx = 0
english_EOS_idx = 2

KIN_PAD_IDX = 0
EN_PAD_IDX = 1

NUM_SPECIAL_TOKENS = 5

MY_PRINTABLE = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


class ParsedToken:
    def __init__(self, w, ffi):
        # POS Info
        self.lm_stem_id = w.lm_stem_id
        self.lm_morph_id = w.lm_morph_id
        self.pos_tag_id = w.pos_tag_id
        self.valid_orthography = w.surface_form_has_valid_orthography

        # Morphology
        self.stem_id = w.stem_id
        self.affix_ids = [w.affix_ids[i] for i in range(w.len_affix_ids)]
        self.extra_stem_token_ids = [w.extra_stem_token_ids[i] for i in range(w.len_extra_stem_token_ids)]

        # Text
        self.uses_bpe = (w.uses_bpe == 1)
        self.is_apostrophed = w.is_apostrophed
        self.surface_form = ffi.string(w.surface_form).decode("utf-8") if (w.len_surface_form > 0) else ''
        self.raw_surface_form = ffi.string(w.raw_surface_form).decode("utf-8") if (w.len_raw_surface_form > 0) else ''
        if (self.is_apostrophed != 0) and (len(self.raw_surface_form) > 0) and (
                (self.raw_surface_form[-1] == 'a') or (self.raw_surface_form[-1] == 'A')):
            self.raw_surface_form = self.raw_surface_form[:-1] + "\'"
        self.syllables: List[Tuple[str, int]] = []
        if len(self.raw_surface_form) > 0:
            seq = text_to_id_sequence(self.raw_surface_form)
            self.syllables.extend([(KINSPEAK_VOCAB_IDX[i], i) for i in seq])
        self.tones: List[Tuple[int, int, int]] = []

    def to_parsed_format(self) -> str:
        word_list = [self.lm_stem_id, self.lm_morph_id, self.pos_tag_id, self.stem_id] + [
            len(self.extra_stem_token_ids)] + self.extra_stem_token_ids + [len(self.affix_ids)] + self.affix_ids
        return ','.join([str(x) for x in word_list])

    def adapted_raw_surface_form(self):
        if self.is_apostrophed:
            return self.raw_surface_form[:-1] + 'a'
        return self.raw_surface_form


COMMA_REPL = 'QWLTIO44O2TORIFHEWEIHGWEIOR094320'
COLON_REPL = '3489YJ12O3H92O20135ITRLW0UTTLEJWL'


class ParsedFlexToken:
    def __init__(self, parsed_token: str, real_parsed_token: ParsedToken = None, syllabify=False):
        self.audio_tokens = []
        if real_parsed_token is not None:
            self.lm_stem_id: int = real_parsed_token.lm_stem_id
            self.lm_morph_id: int = real_parsed_token.lm_morph_id
            self.pos_tag_id: int = real_parsed_token.pos_tag_id
            self.id_stem: int = real_parsed_token.stem_id
            self.id_extra_tokens: List[int] = real_parsed_token.extra_stem_token_ids
            self.stems_ids: List[int] = [self.id_stem] + self.id_extra_tokens
            self.affixes: List[int] = real_parsed_token.affix_ids
            self.is_apostrophed: bool = real_parsed_token.is_apostrophed
            self.surface_form: str = real_parsed_token.surface_form
            self.raw_surface_form: str = real_parsed_token.raw_surface_form
            self.syllables: List[Tuple[str, int]] = real_parsed_token.syllables
            self.tones: List[Tuple[int, int, int]] = real_parsed_token.tones
        else:
            self.surface_form: str = '_'
            self.raw_surface_form: str = '_'
            self.syllables: List[Tuple[str, int]] = []
            self.tones: List[Tuple[int, int, int]] = []
            pieces = parsed_token.split(' ')
            morpho_items = pieces[0].split(',')
            if len(pieces) > 1:
                self.raw_surface_form = pieces[1]
                self.surface_form = self.raw_surface_form.lower()
                if len(pieces) > 2:
                    syls: List[List[str]] = [tks.replace('::', COLON_REPL + ':').split(':') for tks in
                                             pieces[2].replace(',:', COMMA_REPL + ':').split(',')]
                    self.syllables.extend(
                        [(':' if (sl[0] == COLON_REPL) else (',' if (sl[0] == COMMA_REPL) else sl[0]), int(sl[1])) for
                         sl in syls])
                    if len(pieces) > 3:
                        tns: List[List[str]] = [tks.split(':') for tks in pieces[3].split(',')]
                        self.tones.extend([(int(tl[0]), int(tl[1]), int(tl[2])) for tl in tns])
                elif syllabify:
                    seq = text_to_id_sequence(self.raw_surface_form.lower())
                    self.syllables.extend([(KINSPEAK_VOCAB_IDX[i], i) for i in seq])
            self.lm_stem_id: int = int(morpho_items[0])
            self.lm_morph_id: int = int(morpho_items[1])
            self.pos_tag_id: int = int(morpho_items[2])
            self.id_stem: int = int(morpho_items[3])
            num_ext = int(morpho_items[4])
            self.id_extra_tokens: List[int] = [int(v) for v in morpho_items[5:(5 + num_ext)]]
            # This is to cap too long tokens for position encoding
            self.id_extra_tokens: List[int] = self.id_extra_tokens[:60]
            self.stems_ids: List[int] = [self.id_stem] + self.id_extra_tokens
            num_afx = int(morpho_items[(5 + num_ext)])
            self.affixes: List[int] = [int(v) for v in morpho_items[(6 + num_ext):(6 + num_ext + num_afx)]]
            self.is_apostrophed: bool = False
        assert ((len(self.affixes) == 0) or (
                len(self.stems_ids) == 1)), f"Extra tokens with affixes: {self.to_parsed_format()}"

    def adapted_raw_surface_form(self):
        if self.is_apostrophed:
            if str(self.raw_surface_form[-1]) == "'":
                return self.raw_surface_form[:-1] + 'a'
        return self.raw_surface_form

    def noun_class_prefix(self, all_affixes):
        for i in self.affixes:
            v = affix_view(i, all_affixes)
            if v.startswith('N:1:'):
                prefix = v.split(':')[-1]
                if prefix == 'n':
                    prefix = 'zi'
                # ["mu", "ba", "mu", "mi", "ri", "ma", "ki", "bi", "n", "n", "ru", "ka", "tu", "bu", "ku", "ha"]
                return prefix
            elif v.startswith('QA:1:'):
                prefix = v.split(':')[-1]
                if prefix == 'n':
                    prefix = 'zi'
                # ["mu", "ba", "mu", "mi", "ri", "ma", "ki", "bi", "n", "n", "ru", "ka", "tu", "bu", "ku", "ha"]
                return prefix
        return None

    def to_parsed_format(self) -> str:
        word_list = [self.lm_stem_id, self.lm_morph_id, self.pos_tag_id, self.id_stem] + [
            len(self.id_extra_tokens)] + self.id_extra_tokens + [len(self.affixes)] + self.affixes
        ret = (','.join([str(x) for x in word_list])) + ' ' + self.raw_surface_form
        if len(self.syllables) > 0:
            ret += (' ' + (','.join([f'{s}:{i}' for s, i in self.syllables])))
            if len(self.tones) > 0:
                ret += (' ' + (','.join([f'{a}:{b}:{c}' for a, b, c in self.tones])))
        return ret


class ParsedAltToken:
    def __init__(self, indices_csv: str, prob: float, surface_form: str):
        self.surface_form: str = surface_form.lower() if (surface_form is not None) else '_'
        self.raw_surface_form: str = surface_form if (surface_form is not None) else '_'
        self.prob: float = prob
        morpho_items = indices_csv.split(',')
        self.lm_stem_id: int = int(morpho_items[0])
        self.lm_morph_id: int = int(morpho_items[1])
        self.pos_tag_id: int = int(morpho_items[2])
        self.id_stem: int = int(morpho_items[3])
        num_ext = int(morpho_items[4])
        self.id_extra_tokens: List[int] = [int(v) for v in morpho_items[5:(5 + num_ext)]]
        # This is to cap too long tokens for position encoding
        self.id_extra_tokens: List[int] = self.id_extra_tokens[:60]
        self.stems_ids: List[int] = [self.id_stem] + self.id_extra_tokens
        num_afx = int(morpho_items[(5 + num_ext)])
        self.affixes: List[int] = [int(v) for v in morpho_items[(6 + num_ext):(6 + num_ext + num_afx)]]

    def __str__(self):
        word_list = [self.lm_stem_id, self.lm_morph_id, self.pos_tag_id, self.id_stem] + [
            len(self.id_extra_tokens)] + self.id_extra_tokens + [len(self.affixes)] + self.affixes
        return (','.join([str(x) for x in word_list])) + f':{self.prob:.4f}'

    def is_multi_token(self) -> bool:
        return len(self.stems_ids) > 1


class ParsedMultiToken:
    def __init__(self, parsed_token_str: str):
        pieces = parsed_token_str.split(' ')
        self.raw_surface_form: str = pieces[1]
        self.alt_tokens: List[ParsedAltToken] = sorted(
            [ParsedAltToken(ip.split(':')[0], float(ip.split(':')[1]), self.raw_surface_form) for ip in
             pieces[0].split('|')], key=lambda x: x.prob, reverse=True)
        # Filter/Remove improbable tokens
        self.alt_tokens = [self.alt_tokens[0]] if self.alt_tokens[0].is_multi_token() else [t for t in self.alt_tokens
                                                                                            if ((
                                                                                                    not t.is_multi_token()) and (
                                                                                                        t.prob > 0.0))]

    def __str__(self):
        return ('|'.join([str(t) for t in self.alt_tokens])) + ' ' + self.raw_surface_form

    def __len__(self):
        return len(self.alt_tokens)


class ParsedSentenceMulti:
    def __init__(self, parsed_sentence_line: str, delimiter='\t'):
        self.multi_tokens: List[ParsedMultiToken] = [ParsedMultiToken(v) for v in parsed_sentence_line.split(delimiter)
                                                     if len(v) > 0]

    def __str__(self):
        return '\t'.join([str(t) for t in self.multi_tokens])

    def __len__(self):
        return len(self.multi_tokens)


class ParsedFlexSentence:
    def __init__(self, parsed_sentence_line: Union[str, None], parsed_tokens: List[ParsedToken] = None,
                 single_flex_parsed_token: Union[ParsedFlexToken, None] = None, delimiter='\t'):
        if single_flex_parsed_token is not None:
            self.tokens: List[ParsedFlexToken] = [single_flex_parsed_token]
        elif parsed_tokens is not None:
            self.tokens: List[ParsedFlexToken] = [ParsedFlexToken('_', real_parsed_token=token) for token in
                                                  parsed_tokens]
        else:
            self.tokens: List[ParsedFlexToken] = [ParsedFlexToken(v) for v in parsed_sentence_line.split(delimiter) if
                                                  len(v) > 0]

    def to_parsed_format(self) -> str:
        return '\t'.join([tk.to_parsed_format() for tk in self.tokens])

    def num_stems(self):
        return len([i for t in self.tokens for i in t.stems_ids])

    def trim(self, max_len):
        while self.num_stems() > max_len:
            self.tokens = self.tokens[:-1]
        return self

    def __len__(self):
        return self.num_stems()


def parse_text_to_morpho_sentence(uds_client: UnixSocketClient, txt: str) -> ParsedFlexSentence:
    parsed_sentence_line = ''
    success = uds_client.send_line('\t' + txt.strip())
    if success:
        parsed_sentence_line = uds_client.read_line()
    return ParsedFlexSentence(parsed_sentence_line)


def parse_document_to_morpho_sentence(uds_client: UnixSocketClient, text_lines: List[str]) -> ParsedFlexSentence:
    ret = ParsedFlexSentence(None, parsed_tokens=[], delimiter='\t')
    for txt in text_lines:
        it = parse_text_to_morpho_sentence(uds_client, txt)
        ret.tokens = ret.tokens + it.tokens
    return ret


special_tokens = ['<pad>', '<unk>', '<msk>', '<s>', '</s>']

STEM_AFSET_CORR_FACTOR_ALPHA = 0.01


class Affix():
    def __init__(self, id, line):
        tk = line.split(':')
        self.id = id
        self.wt = int(tk[0])
        self.slot = int(tk[1])
        self.idx = int(tk[2])
        self.key = ':'.join(tk[:3])
        self.prob = float(tk[3])
        try:
            self.view = all_wt_abbrevs[self.wt] + ':' + str(self.slot) + ':' + \
                        all_word_types[self.wt]["morpheme_sets"][self.slot][self.idx]
        except Exception as e:
            raise Exception(
                f'Affix Error with input line: {line} ==> id: {self.id}, wt: {self.wt}, slot: {self.slot}, idx: {self.idx}, key: {self.key}, prob: {self.prob}')


class Afset():
    def __init__(self, id, line):
        tk = line.split(':')
        self.id = id
        self.wt = int(tk[0])
        self.fsa = int(tk[1])
        self.len_indices = int(tk[2])
        self.indices = [int(s) for s in tk[3].split(',')]
        self.prob = float(tk[4])
        try:
            self.slots = all_word_types[self.wt]["morphotactics_fsa"][self.fsa]
            self.affixes_keys = [(str(self.wt) + ':' + str(s) + ':' + str(i)) for s, i in zip(self.slots, self.indices)
                                 if (
                                         (all_word_types[self.wt]["stem_start_idx"] != s) and (
                                         all_word_types[self.wt]["stem_end_idx"] != s))]
            self.view = all_wt_abbrevs[self.wt] + ':' + '-'.join([(all_word_types[self.wt]["morpheme_sets"][s][i]) if (
                    (all_word_types[self.wt]["stem_start_idx"] != s) and (
                    all_word_types[self.wt]["stem_end_idx"] != s)) else "*" for s, i in
                                                                  zip(self.slots, self.indices)])
        except Exception as e:
            raise Exception(
                f'Afset Error with input line: {line} ==> id: {self.id}, wt: {self.wt}, fsa: {self.fsa}, len_indices: {self.len_indices}, indices: {self.indices}, prob: {self.prob}') from e


def read_all_affixes(fn):
    with open(fn) as f:
        affixes = [Affix(i, line.rstrip('\n')) for i, line in enumerate(f) if len(line.rstrip('\n')) > 0]
    return affixes


def read_corr_table(fn):
    ret = dict()
    with open(fn) as f:
        for line in f:
            ln = line.rstrip('\n')
            if len(ln) > 0:
                t = ln.split('\t')
                ret[t[0]] = float(t[1])
    return ret


def read_all_afsets(fn):
    with open(fn) as f:
        afsets = [Afset(i, line.rstrip('\n')) for i, line in enumerate(f) if len(line.rstrip('\n')) > 0]
    return afsets


def affix_view(id, all_affixes):
    if id < 5:
        return special_tokens[id]
    else:
        return all_affixes[id - 5].view


def id_to_affix(id, all_affixes) -> Union[Affix, None]:
    if id < 5:
        return None
    else:
        return all_affixes[id - 5]


morph_pos_tags = 0


def afset_view(id, all_afsets):
    global morph_pos_tags
    if id < 5:
        return special_tokens[id]
    elif (id - 5) < len(all_afsets):
        return all_afsets[id - 5].view
    else:
        if morph_pos_tags == 0:
            morph_pos_tags = len([p for p in all_pos_tags if p["type"] == 0])
        if (id - len(all_afsets) + morph_pos_tags - 5) < len(all_pos_tags):
            return pos_tag_view(id - len(all_afsets) + morph_pos_tags - 1)
        else:
            return f'[*unk*:{id}]'


def id_to_afset(id, all_afsets) -> Union[Afset, None]:
    global morph_pos_tags
    if id < 5:
        return None
    elif (id - 5) < len(all_afsets):
        return all_afsets[id - 5]
    else:
        return None


def pos_tag_view(id):
    if id < 5:
        return special_tokens[id]
    elif (id - 5) < len(all_pos_tags):
        return all_pos_tags[id - 5]["name"] + '#' + '{:03d}'.format(all_pos_tags[id - 5]["idx"])
    else:
        return f'[*unk*:{id}]'


def pos_tag_initials(id):
    if id < 5:
        return special_tokens[id]
    elif (id - 5) < len(all_pos_tags):
        return all_pos_tags[id - 5]["name"]
    else:
        return '<rare>'


def synth_morpho_token_via_socket(uds_client: UnixSocketClient, wt_idx: int, stem: str, fsa_key: str,
                                  indices_csv: str) -> str:
    request: str = f"|{wt_idx}|{stem}|{fsa_key}|{indices_csv}\n"
    success = uds_client.send_line(request)
    response = ''
    if success:
        response = uds_client.read_line()
    return response


def make_surface_form(stem_id, affix_ids, stems_vocab, all_affixes, uds_client: UnixSocketClient, debug=False,
                      retry=False) -> Tuple[
    str, Optional[str], bool]:
    if (len(affix_ids) > 0):
        stem = stems_vocab[stem_id]
        if stem.find(':') > 0:
            stem = stem.split(':')[1]
        affixes_list = [id_to_affix(id, all_affixes) for id in affix_ids]
        wt_idx = affixes_list[0].wt
        stem_slot = all_word_types[wt_idx]['stem_start_idx']
        slots_idx = [(x.slot, x.idx) for x in affixes_list]
        slots = set([x.slot for x in affixes_list])
        stem_idx = 0
        if wt_idx > 1:
            arr = [i for i, v in enumerate(all_word_types[wt_idx]['morpheme_sets'][stem_slot]) if (v == stem)]
            if len(arr) > 0:
                stem_idx = arr[0]
        if stem_slot not in slots:
            slots_idx = slots_idx + [(stem_slot, stem_idx)]  # bug: not all stem slot idx=0
        # Handle verb reduplication
        if (wt_idx == 0) and (12 in slots) and (11 not in slots):
            slots_idx = slots_idx + [(11, 0)]
        slots_idx = sorted(slots_idx, key=lambda x: x[0], reverse=False)

        fsa_key = str(wt_idx) + ':' + '-'.join([str(x[0]) for x in slots_idx])
        indices_csv = ','.join([str(x[1]) for x in slots_idx])

        try:
            pseudo_str = ('-' if debug else '').join(
                [stem if (sl == stem_slot) else (all_word_types[wt_idx]['morpheme_sets'][sl][ix]) for sl, ix in
                 slots_idx])
        except IndexError as err:
            pseudo_str = stem + '+' + ('-'.join([a.view for a in affixes_list]))
            # print(f'Index error at: wt_idx: {wt_idx}, stem_id: {stem_id}, stem: \'{stem}\'  slots_idx: {slots_idx}, affix_ids: {affix_ids}, affixes_list: {", ".join([a.view for a in affixes_list])}')
            # raise err

        if (uds_client is None):
            ret_str = ('-'.join([affix_view(af.id + 5, all_affixes) for af in
                                 affixes_list]) + '/' + stem + "({},'{}','{}','{}')/{}".format(wt_idx, stem, fsa_key,
                                                                                               indices_csv,
                                                                                               pseudo_str)) if debug else pseudo_str
            return ret_str, None, False
        else:
            surface_form = synth_morpho_token_via_socket(uds_client, wt_idx, stem, fsa_key, indices_csv)
            if not any(c in MY_PRINTABLE for c in surface_form):
                ret_str = ('-'.join([affix_view(af.id + 5, all_affixes) for af in
                                     affixes_list]) + '/' + stem + "({},'{}','{}','{}')/{}/{}".format(wt_idx, stem,
                                                                                                      fsa_key,
                                                                                                      indices_csv,
                                                                                                      surface_form,
                                                                                                      pseudo_str)) if debug else pseudo_str
                return ret_str, None, False
            else:
                # The returned form is already in parsed format.
                parts = surface_form.split()
                return parts[1], surface_form, True
    else:
        surface_form = stems_vocab[stem_id]
        if surface_form is not None:
            if (':' in surface_form) and (len(surface_form) > 2):
                cdx = surface_form.index(':')
                if cdx < (len(surface_form) - 1):
                    surface_form = surface_form[(cdx + 1):]
        return surface_form, None, True


def decode_word_per_wt(top_stems, top_pos_tags, top_afsets, top_affixes,
                       top_stems_prob, top_pos_tags_prob, top_afsets_prob, top_affixes_prob,
                       stems_vocab, all_afsets, all_affixes, all_afsets_inverted_index,
                       pos_afset_corr, pos_stem_corr, afset_stem_corr, afset_affix_slot_corr,
                       uds_client, wt, wt_prob,
                       prob_cutoff=0.3, affix_prob_cutoff=0.3, affix_min_prob=0.3,
                       debug=False,
                       retry=False):
    # 2. wt filtering
    stems = []
    pos_tags = []
    afsets = []
    affixes = []

    stems_prob = []
    pos_tags_prob = []
    afsets_prob = []
    affixes_prob = []

    for id, p in zip(top_stems, top_stems_prob):
        if stems_vocab[id].find(':') > 0:
            t = all_wt_abbrevs_dict[stems_vocab[id].split(':')[0]]
            if (wt == t):
                stems.append(id)
                stems_prob.append(p)
        elif (wt == -1):
            stems.append(id)
            stems_prob.append(p)

    for id, p in zip(top_pos_tags, top_pos_tags_prob):
        if (id >= 5) and ((id - 5) < len(all_pos_tags)):
            nm = all_pos_tags[id - 5]["name"]
            if nm in all_wt_abbrevs_dict:
                t = all_wt_abbrevs_dict[nm]
                if (wt == t):
                    pos_tags.append(id)
                    pos_tags_prob.append(p)
            elif (wt == -1):
                pos_tags.append(id)
                pos_tags_prob.append(p)
        elif (wt == -1):
            pos_tags.append(id)
            pos_tags_prob.append(p)

    for id, p in zip(top_afsets, top_afsets_prob):
        af = id_to_afset(id, all_afsets)
        if af is not None:
            if (wt == af.wt):
                afsets.append(id)
                afsets_prob.append(p)
        elif (wt == -1):
            afsets.append(id)
            afsets_prob.append(p)

    for id, p in zip(top_affixes, top_affixes_prob):
        af = id_to_affix(id, all_affixes)
        if af is not None:
            if (wt == af.wt):
                affixes.append(id)
                affixes_prob.append(p)

    # 3. prob cut-off
    stems_cut = []
    pos_tags_cut = []
    afsets_cut = []
    affixes_cut = []

    stems_prob_cut = []
    pos_tags_prob_cut = []
    afsets_prob_cut = []
    affixes_prob_cut = []

    i = 0
    pp = 0.0
    tt = 0
    for id, p in zip(stems, stems_prob):
        drop = (pp - p)
        if (i > 0) and ((drop >= prob_cutoff) or (drop > (1.0 - tt))):
            break
        stems_cut.append(id)
        stems_prob_cut.append(p)
        #         print(f'STEM: {id} :> {p} :> {math.log(p+1e-50)}')
        # if debug:
        #     print('cut-stem:', stems_vocab[id], f'{p:.4f}')
        i += 1
        pp = p
        tt += p

    i = 0
    pp = 0.0
    tt = 0
    for id, p in zip(pos_tags, pos_tags_prob):
        drop = (pp - p)
        if (i > 0) and ((drop >= prob_cutoff) or (drop > (1.0 - tt))):
            break
        pos_tags_cut.append(id)
        pos_tags_prob_cut.append(p)
        # if debug:
        #     print('cut-pos_tag:', pos_tag_view(id), f'{p:.4f}')
        i += 1
        pp = p
        tt += p

    i = 0
    pp = 0.0
    tt = 0.0
    for id, p in zip(afsets, afsets_prob):
        drop = (pp - p)
        if (i > 0) and ((drop >= prob_cutoff) or (drop > (1.0 - tt))):
            break
        afsets_cut.append(id)
        afsets_prob_cut.append(p)
        # if debug:
        #     print('cut-afset:', afset_view(id, all_afsets), f'{p:.4f}')
        i += 1
        pp = p
        tt += p

    i = 0
    pp = 0.0
    for id, p in zip(affixes, affixes_prob):
        if ((i > 0) and ((pp - p) >= affix_prob_cutoff)) or (p < affix_min_prob):
            break
        affixes_cut.append(id)
        affixes_prob_cut.append(p)
        # if debug:
        #     print('cut-affix:', affix_view(id, all_affixes), f'{p:.4f}')
        i += 1
        if affix_view(id, all_affixes)[-1] != '*':
            pp = p

    affixes_cut_set = set(affixes_cut)

    # 5. fsa filtering
    afset_own_affixes = dict()
    for idx_afset, afs_id in enumerate(afsets_cut):
        non_afset_affixes_stats = 0.0
        afset_affixes = []
        afset_affixes_slots = set()
        afs = id_to_afset(afs_id, all_afsets)
        if afs is not None:
            for k in afs.affixes_keys:
                afx = all_afsets_inverted_index[k]
                if afx.slot not in afset_affixes_slots:
                    myafid = afx.id + NUM_SPECIAL_TOKENS
                    if not (myafid in affixes_cut_set):
                        afsets_prob_cut[idx_afset] = 0.0
                    afset_affixes.append(myafid)
                    afset_affixes_slots.add(afx.slot)
        # affix slot conflict resolution
        for a, p in zip(affixes_cut, affixes_prob_cut):
            if p >= affix_min_prob:
                afx = id_to_affix(a, all_affixes)
                if afx is None:
                    print('Can\'t find affix:', affix_view(a, all_affixes))
                if afx is not None:
                    af_key = '{}-{}:{}'.format(afs_id, afx.wt, afx.slot)
                    if (afx.slot not in afset_affixes_slots) and (af_key in afset_affix_slot_corr):
                        if afsets_prob_cut[idx_afset] > 0.0:
                            afsets_prob_cut[idx_afset] += p
                        afset_affixes.append(a)
                        afset_affixes_slots.add(afx.slot)
                        non_afset_affixes_stats += 1.0
        afset_own_affixes[afs_id] = afset_affixes, non_afset_affixes_stats
        # if debug:
        #     print('afset-affixes:', '@', afset_view(afs_id, all_afsets), '==>', ', '.join([affix_view(i,all_affixes) for i in afset_affixes]))

    # 5. Apply correlations
    corr = np.zeros((len(stems_cut), len(pos_tags_cut), len(afsets_cut)))  # + 1e-7

    for si, s in enumerate(stems_cut):
        for pi, p in enumerate(pos_tags_cut):
            for ai, a in enumerate(afsets_cut):
                ps_key = '{}-{}'.format(p, s)
                pa_key = '{}-{}'.format(p, a)
                as_key = '{}-{}'.format(a, s)
                if (ps_key in pos_stem_corr) and (pa_key in pos_afset_corr) and (as_key in afset_stem_corr):
                    corr[si, pi, ai] = max(corr[si, pi, ai],
                                           math.exp(afset_stem_corr[as_key] + pos_stem_corr[ps_key] + pos_afset_corr[
                                               pa_key]))

    results = {}
    corr_Z = corr.sum()
    for si, s in enumerate(stems_cut):
        for pi, p in enumerate(pos_tags_cut):
            for ai, a in enumerate(afsets_cut):
                results[(s, p, a)] = ((corr[si, pi, ai] / corr_Z) if (corr_Z > 0.0) else 0.0)

    results_list = [(k, v) for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)]

    results_list = sorted(results_list, key=lambda x: x[1], reverse=True)

    ret_stems = [s for (s, p, a), prob in results_list]
    ret_pos_tags = [p for (s, p, a), prob in results_list]
    ret_afsets = [a for (s, p, a), prob in results_list]
    ret_affixes = [afset_own_affixes[a][0] for (s, p, a), prob in results_list]
    ret_non_afset_affixes_stats_list = [afset_own_affixes[a][1] for (s, p, a), prob in results_list]

    ret_probs = [(prob) for (s, p, a), prob in results_list]

    surface_forms = [
        make_surface_form(s, ret_affixes[i], stems_vocab, all_affixes, uds_client, debug=debug, retry=retry)
        for i, ((s, p, a), prob)
        in enumerate(results_list)]

    gen_word = surface_forms[0][0] if len(surface_forms) > 0 else '<none>'
    for i, (wrd, parse, flg) in enumerate(surface_forms):
        if flg:
            gen_word = wrd  # + ' ({})'.format(pos_tag_view(ret_pos_tags[i]))
            break

    return (gen_word,
            surface_forms,
            ret_stems,
            ret_pos_tags,
            ret_afsets,
            ret_affixes,
            ret_probs,
            ret_non_afset_affixes_stats_list)


def decode_bpe_word_per_wt(top_stems: List[List[int]], top_pos_tags, top_afsets,
                           top_stems_prob, top_pos_tags_prob, top_afsets_prob,
                           stems_vocab,
                           wt_prob,
                           pos_afset_corr, pos_stem_corr, afset_stem_corr,
                           prob_cutoff=0.3):
    # 2. wt filtering
    stems: List[List[int]] = []
    pos_tags = []
    afsets = []

    stems_prob = []
    pos_tags_prob = []
    afsets_prob = []

    for ids, p in zip(top_stems, top_stems_prob):
        stems.append(ids)
        stems_prob.append(p)

    for id, p in zip(top_pos_tags, top_pos_tags_prob):
        pos_tags.append(id)
        pos_tags_prob.append(p)

    for id, p in zip(top_afsets, top_afsets_prob):
        afsets.append(id)
        afsets_prob.append(p)

    # 3. prob cut-off
    stems_cut: List[List[int]] = []
    pos_tags_cut = []
    afsets_cut = []

    stems_prob_cut = []
    pos_tags_prob_cut = []
    afsets_prob_cut = []

    i = 0
    pp = 0.0
    for ids, p in zip(stems, stems_prob):
        if (i > 0) and ((pp - p) >= prob_cutoff):
            break
        stems_cut.append(ids)
        stems_prob_cut.append(p)
        #         print(f'STEM: {id} :> {p} :> {math.log(p+1e-50)}')
        i += 1
        pp = p

    i = 0
    pp = 0.0
    for id, p in zip(pos_tags, pos_tags_prob):
        if (i > 0) and ((pp - p) >= prob_cutoff):
            break
        pos_tags_cut.append(id)
        pos_tags_prob_cut.append(p)
        i += 1
        pp = p

    i = 0
    pp = 0.0
    for id, p in zip(afsets, afsets_prob):
        if (i > 0) and ((pp - p) >= prob_cutoff):
            break
        afsets_cut.append(id)
        afsets_prob_cut.append(p)
        i += 1
        pp = p

    # 5. Apply correlations
    corr = np.zeros((len(stems_cut), len(pos_tags_cut), len(afsets_cut)))  # + 1e-7

    for si, stids in enumerate(stems_cut):
        for pi, p in enumerate(pos_tags_cut):
            for ai, a in enumerate(afsets_cut):
                ps_key = '{}-{}'.format(p, stids[0])
                pa_key = '{}-{}'.format(p, a)
                as_key = '{}-{}'.format(a, stids[0])
                if (ps_key in pos_stem_corr) and (pa_key in pos_afset_corr) and (as_key in afset_stem_corr):
                    corr[si, pi, ai] = max(corr[si, pi, ai],
                                           math.exp((afset_stem_corr[as_key] * STEM_AFSET_CORR_FACTOR_ALPHA) +
                                                    math.log(stems_prob_cut[si] + 1e-50) +
                                                    math.log(pos_tags_prob_cut[pi] + 1e-50) +
                                                    math.log(afsets_prob_cut[ai] + 1e-50)))

    results = {}
    corr_Z = corr.sum()
    for si, sids in enumerate(stems_cut):
        for pi, p in enumerate(pos_tags_cut):
            for ai, a in enumerate(afsets_cut):
                st = ','.join([f'{i}' for i in sids])
                results[(st, p, a)] = ((corr[si, pi, ai] / corr_Z) if (corr_Z != 0.0) else 0.0)

    results_list = [(k, v) for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)]

    ret_stems = [[int(i) for i in s.split(',')] for (s, p, a), prob in results_list]
    ret_pos_tags = [p for (s, p, a), prob in results_list]
    ret_afsets = [a for (s, p, a), prob in results_list]
    ret_probs = [(prob * wt_prob) for (s, p, a), prob in results_list]

    surface_forms = []
    for stems in ret_stems:
        sforms = []
        for i in stems:
            s = stems_vocab[i]
            if s[0] == '▁':
                s = s[1:]
            elif s.startswith('@@'):
                s = s[2:]
            sforms.append(s)
        st = str(''.join(sforms))
        surface_forms.append((st, len(st) > 0))
    gen_word = surface_forms[0][0] if len(surface_forms) > 0 else 'N/A'
    return (gen_word,
            surface_forms,
            ret_stems,
            ret_pos_tags,
            ret_afsets,
            ret_probs)


def decode_word(top_stems, top_pos_tags, top_afsets, top_affixes,
                top_stems_prob, top_pos_tags_prob, top_afsets_prob, top_affixes_prob,
                stems_vocab, all_afsets, all_affixes, all_afsets_inverted_index,
                pos_afset_corr, pos_stem_corr, afset_stem_corr, afset_affix_slot_corr,
                uds_client,
                prob_cutoff=0.3, affix_prob_cutoff=0.3,
                affix_min_prob=0.3, lprob_score_delta=2.0,
                debug=False,
                retry=False):
    # 1. wt resolution --> morpho_wt, -1: other
    wt_list = [i for i in range(len(all_wt_abbrevs))] + [-1]
    wt_votes = [0.0 for _ in range(len(wt_list))]
    wt_vals = [-100.0 for _ in range(len(wt_list))]

    for id, p in zip(top_stems, top_stems_prob):
        if (stems_vocab[id].find(':') > 0) and (len(stems_vocab[id]) > 1):
            t = all_wt_abbrevs_dict[stems_vocab[id].split(':')[0]]
            wt_vals[t] = max(wt_vals[t], math.log(p + 1e-50))
        else:
            wt_vals[-1] = max(wt_vals[-1], math.log(p + 1e-50))

    wt_votes = [a + b for a, b in zip(wt_votes, wt_vals)]

    wt_vals = [-100.0 for _ in range(len(wt_list))]
    for id, p in zip(top_pos_tags, top_pos_tags_prob):
        if (id >= 5) and ((id - 5) < len(all_pos_tags)):
            nm = all_pos_tags[id - 5]["name"]
            if nm in all_wt_abbrevs_dict:
                t = all_wt_abbrevs_dict[nm]
                wt_vals[t] = max(wt_vals[t], math.log(p + 1e-50))
            else:
                wt_vals[-1] = max(wt_vals[-1], math.log(p + 1e-50))
        else:
            wt_vals[-1] = max(wt_vals[-1], math.log(p + 1e-50))

    wt_votes = [a + b for a, b in zip(wt_votes, wt_vals)]

    wt_vals = [-100.0 for _ in range(len(wt_list))]
    for id, p in zip(top_afsets, top_afsets_prob):
        af = id_to_afset(id, all_afsets)
        if af is not None:
            t = af.wt
            wt_vals[t] = max(wt_vals[t], math.log(p + 1e-50))
        else:
            wt_vals[-1] = max(wt_vals[-1], math.log(p + 1e-50))

    wt_votes = [a + b for a, b in zip(wt_votes, wt_vals)]

    wt_Z = sum([math.exp(v) for v in wt_votes])
    wt_probs = [(math.exp(v) / wt_Z) if (wt_Z != 0.0) else 0.0 for v in wt_votes]

    wt_tuples = sorted([(id, sc, pr) for (id, sc, pr) in zip(wt_list, wt_votes, wt_probs)], key=lambda x: x[1],
                       reverse=True)

    return_list = []
    prev_wt_score = wt_tuples[0][1]
    for wt, wt_score, wt_prob in wt_tuples:
        if (prev_wt_score - wt_score) > lprob_score_delta:
            break
        ret = decode_word_per_wt(top_stems, top_pos_tags, top_afsets, top_affixes,
                                 top_stems_prob, top_pos_tags_prob, top_afsets_prob, top_affixes_prob,
                                 stems_vocab, all_afsets, all_affixes, all_afsets_inverted_index,
                                 pos_afset_corr, pos_stem_corr, afset_stem_corr, afset_affix_slot_corr,
                                 uds_client, wt, wt_prob,
                                 prob_cutoff=prob_cutoff, affix_prob_cutoff=affix_prob_cutoff,
                                 affix_min_prob=affix_min_prob,
                                 debug=debug,
                                 retry=retry)
        return_list.append(ret)
        prev_wt_score = wt_score
    return return_list


def decode_bpe_word(top_stems, top_pos_tags, top_afsets,
                    top_stems_prob, top_pos_tags_prob, top_afsets_prob,
                    stems_vocab, all_afsets,
                    pos_afset_corr, pos_stem_corr, afset_stem_corr,
                    prob_cutoff=0.3, lprob_score_delta=2.0):
    # 1. wt resolution --> morpho_wt, -1: other
    wt_list = [-1]
    wt_votes = [0.0 for _ in range(len(wt_list))]
    wt_vals = [-100.0 for _ in range(len(wt_list))]

    for ids, p in zip(top_stems, top_stems_prob):
        wt_vals[-1] = max(wt_vals[-1], math.log(p + 1e-50))

    wt_votes = [a + b for a, b in zip(wt_votes, wt_vals)]

    wt_vals = [-100.0 for _ in range(len(wt_list))]
    for id, p in zip(top_pos_tags, top_pos_tags_prob):
        if not ((id >= 5) and ((id - 5) < len(all_pos_tags))):
            wt_vals[-1] = max(wt_vals[-1], math.log(p + 1e-50))

    wt_votes = [a + b for a, b in zip(wt_votes, wt_vals)]

    wt_vals = [-100.0 for _ in range(len(wt_list))]
    for id, p in zip(top_afsets, top_afsets_prob):
        af = id_to_afset(id, all_afsets)
        if af is None:
            wt_vals[-1] = max(wt_vals[-1], math.log(p + 1e-50))

    wt_votes = [a + b for a, b in zip(wt_votes, wt_vals)]

    wt_Z = sum([math.exp(v) for v in wt_votes])
    wt_probs = [(math.exp(v) / wt_Z) if (wt_Z != 0.0) else 0.0 for v in wt_votes]

    wt_tuples = sorted([(id, sc, pr) for (id, sc, pr) in zip(wt_list, wt_votes, wt_probs)], key=lambda x: x[1],
                       reverse=True)

    return_list = []
    prev_wt_score = wt_tuples[0][1]
    for wt, wt_score, wt_prob in wt_tuples:
        if ((prev_wt_score - wt_score) > lprob_score_delta) and (len(return_list) > 0):
            break
        ret = decode_bpe_word_per_wt(top_stems, top_pos_tags, top_afsets,
                                     top_stems_prob, top_pos_tags_prob, top_afsets_prob,
                                     stems_vocab, wt_prob, pos_afset_corr, pos_stem_corr, afset_stem_corr,
                                     prob_cutoff=prob_cutoff)
        if len(ret[1]) > 0:
            return_list.append(ret)
        prev_wt_score = wt_score
    return return_list


class Kinyarwanda:
    def __init__(self, affixes_prob_file="affixes_prob_file_2024-05-01.txt",
                 pronunciation_adapter_file="agai_pronunciation_adapter.tsv"):
        # ########################################################################
        # From symbols data
        self._pad = "_"
        self._punctuation = [";", ":", ",", ".", "!", "?", "¡", "¿", "—", "…", "\"", "«", "»", "“", "”", "\'", " "]
        self._letters = ["i", "u", "o", "a", "e", "b", "c", "d", "f", "g", "h", "j", "k", "m", "n", "p", "r", "l", "s",
                         "t", "v", "y", "w", "z", "bw", "by", "cw", "cy", "dw", "fw", "gw", "hw", "kw", "jw", "jy",
                         "ny", "mw", "my", "nw", "pw", "py", "rw", "ry", "sw", "sy", "tw", "ty", "vw", "vy", "zw", "pf",
                         "ts", "sh", "shy", "mp", "mb", "mf", "mv", "nc", "nj", "nk", "ng", "nt", "nd", "ns", "nz",
                         "nny", "nyw", "byw", "ryw", "shw", "tsw", "pfy", "mbw", "mby", "mfw", "mpw", "mpy", "mvw",
                         "mvy", "myw", "ncw", "ncy", "nsh", "ndw", "ndy", "njw", "njy", "nkw", "ngw", "nsw", "nsy",
                         "ntw", "nty", "nzw", "shyw", "mbyw", "mvyw", "nshy", "nshw", "nshyw", "bg", "pfw", "pfyw",
                         "vyw", "njyw", "x", "q"]

        # Export all symbols:
        self.tts_symbols = [self._pad] + self._punctuation + self._letters

        # Special symbol ids
        self.SPACE_ID = self.tts_symbols.index(" ")

        # ########################################################################

        # Mappings from symbol to numeric ID and vice versa:
        self._symbol_to_id = {s: i for i, s in enumerate(self.tts_symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.tts_symbols)}

        self._digits_map = {0: "zeru",
                            1: "rimwe",
                            2: "kabiri",
                            3: "gatatu",
                            4: "kane",
                            5: "gatanu",
                            6: "gatandatu",
                            7: "karindwi",
                            8: "umunani",
                            9: "icyenda"}

        self._rw_phone_pattern = r"(((\+250)|(250)|(0))?7[0-9]\d{7})"

        self._VOWELS = {"i", "u", "o", "a", "e"}
        self._CONSONANTS = {"b", "c", "d", "f", "g", "h", "j", "k", "m", "n", "p", "r", "l", "s", "t", "v", "y", "w",
                            "z", "bw", "by", "cw", "cy", "dw", "fw", "gw", "hw", "kw", "jw", "jy", "ny", "mw", "my",
                            "nw", "pw", "py", "rw", "ry", "sw", "sy", "tw", "ty", "vw", "vy", "zw", "pf", "ts", "sh",
                            "shy", "mp", "mb", "mf", "mv", "nc", "nj", "nk", "ng", "nt", "nd", "ns", "nz", "nny", "nyw",
                            "byw", "ryw", "shw", "tsw", "pfy", "mbw", "mby", "mfw", "mpw", "mpy", "mvw", "mvy", "myw",
                            "ncw", "ncy", "nsh", "ndw", "ndy", "njw", "njy", "nkw", "ngw", "nsw", "nsy", "ntw", "nty",
                            "nzw", "shyw", "mbyw", "mvyw", "nshy", "nshw", "nshyw", "bg", "pfw", "pfyw", "vyw", "njyw"}

        self.hour_map = {0: "sita z'ijoro", 1: "saba", 2: "munani", 3: "cyenda", 4: "kumi", 5: "kumi n'imwe",
                         6: "kumi n'ebyiri",
                         7: "moya", 8: "mbiri", 9: "tatu", 10: "yine", 11: "tanu", 12: "sita", 13: "saba", 14: "munani",
                         15: "cyenda", 16: "kumi", 17: "kumi n'imwe", 18: "kumi n'ebyiri", 19: "moya", 20: "mbiri",
                         21: "tatu",
                         22: "yine", 23: "tanu", 24: "sita z'ijoro"}

        # mu ba mu mi ri ma ki bi n n ru ka tu bu ku ha ku
        self.Rwanda_Months = {"mutarama", "gashyantare", "werurwe", "mata", "gicurasi", "kamena", "nyakanga", "kanama",
                              "nzeli", "nzeri", "ukwakira", "ugushyingo", "ukuboza"}

        self.time_regex = "^([01]?[0-9]|2[0-3])(:[0-5]?[0-9]){1,2}$"
        self.time_pattern = re.compile(self.time_regex)

        self.class_prefixes = {"mu", "ba", "mi", "ri", "ma", "ki", "bi", "n", "ru", "ka", "tu", "bu", "ku", "ha"}

        self.all_affixes = []
        if os.path.exists(affixes_prob_file):
            self.all_affixes = read_all_affixes(affixes_prob_file)

        self.token_map: Dict[str, List[str]] = {
            "%": ["ku", "ijana"],
            "cm": ["santimetero"],
            "dm": ["desimetero"],
            "m": ["metero"],
            "kg": ["ibiro"],
            "g": ["garama"],
            "mg": ["miligarama"],
            "t": ["toni"],
            "oc": ["dogere", "selisiyusi"],
            "canada": ["kanada"],
            "congo": ["kongo"],
            "america": ["amerika"],
            "france": ["faranse"],
            "antrachnose": ["antarakinoze"],
            "ph": ["pe", "hashi"],
            "rab": ["rabu"],
            "managri": ["minagiri"],
            "kcl": ["potasiyamu", "kolorayide"]
        }
        if os.path.exists(pronunciation_adapter_file):
            with open(pronunciation_adapter_file, "r") as tsv:
                for line in tsv:
                    tokens = line.rstrip("\n").split("\t")
                    if len(tokens) == 2:
                        if tokens[0] not in self.token_map:
                            self.token_map[tokens[0]] = " ".join(tokens[1].split()).split()

    def process_cons(self, cons, seq):
        if cons in self._symbol_to_id:
            seq.append(self._symbol_to_id[cons])
        else:
            for c in cons:
                if c in self._symbol_to_id:
                    seq.append(self._symbol_to_id[c])

    def rw_prefix(self, word: str, n: int, pos="", prefix=None):
        if (len(word) < 4) or (word.lower() in self.Rwanda_Months):
            return None
        if len(word) == 0:
            return "bi"
        if (word == "biro") or (word == "ibiro"):
            return "bi"
        if (word == "minsi") or (word == "iminsi"):
            return "mi"
        if (word[0] == "n") or (word == "hegitari") or (word == "ari") or (word == "ali") or (word == "are") or (
                word == "santimetero") or (word == "selisiusi") or (word == "selisiyusi") or (word == "kilometero") or (
                word == "dogere") or (word == "kirometero") or word.endswith("litiro") or word.endswith(
            "ritiro") or word.endswith("metero") or (word == "ha") or (
                word == "cm") or (word == "m") or (word == "toni") or (word == "mg") or (word == "garama") or (
                word == "miligarama") or (word == "mirigarama") or (pos == "UN"):
            return "zi"
        if prefix is not None:
            return prefix
        if word[0] in self._VOWELS:
            return self.rw_prefix(word[1:], n, pos=pos)
        if (word[0:2] == "cy") and (word != "cyangwa"):
            return "ki"
        if word[1] == "w":
            return word[0:1] + "u"
        if (word[1] == "y") and (word != "cyangwa"):
            return word[0:1] + "i"
        pr = word[0:2]
        if (pr == "du"):
            pr = "tu"
        if (pr == "ga"):
            pr = "ka"
        if (pr == "gi"):
            pr = "ki"
        if (pr == "gu"):
            pr = "ku"
        if pr in self.class_prefixes:
            return pr
        return "ri" if (n == 1) else ("ka" if (n < 8) else None)

    def is_time(self, tok):
        if tok == "":
            return False
        return re.search(self.time_pattern, tok) is not None

    def get_phone_pieces(self, phone):
        phone_pieces = []
        if phone[0] == "+":
            phone_pieces.append("+")
            left = phone[1:]
        else:
            left = phone

        if left.startswith("250"):
            phone_pieces.append("250")
            left = left[3:]

        if left.startswith("7"):
            phone_pieces.append("7")
            left = left[1:]
        elif left.startswith("07"):
            phone_pieces.append("0")
            phone_pieces.append("7")
            left = left[2:]
        if (len(left) % 2) == 0:
            for i in range(0, len(left), 2):
                pc = left[i:i + 2]
                if pc[0] == "0":
                    phone_pieces.append("0")
                    phone_pieces.append(pc[1])
                else:
                    phone_pieces.append(pc)
        elif len(left) > 1:
            phone_pieces.append(left[0])
            left = left[1:]
            for i in range(0, len(left), 2):
                pc = left[i:i + 2]
                if pc[0] == "0":
                    phone_pieces.append("0")
                    phone_pieces.append(pc[1])
                else:
                    phone_pieces.append(pc)
        elif len(left) == 1:
            phone_pieces.append(left)
        return phone_pieces

    def spell_time(self, time) -> List[str]:
        try:
            times = [int(t) for t in time.split(":")]
            hr = 0 if len(times) < 1 else times[0]
            mn = 0 if len(times) < 2 else times[1]
            sec = 0 if len(times) < 3 else times[2]
            minutes_list = [] if (mn == 0) else (
                ["n'umunota", "umwe"] if (mn == 1) else ["n'iminota", rw_spell_number("mi", mn)])
            seconds_list = [] if (sec == 0) else (
                ["n'isegonda", "rimwe"] if (sec == 1) else ["n'amasegonda", rw_spell_number("ma", sec)])
            return [self.hour_map[hr]] + minutes_list + seconds_list
        except:
            pass
        return [time]

    def txt2seq(self, txt):
        seq = []
        txt = re.sub(r"\s+", "|", txt).strip()
        start = 0
        end = 0
        while end < len(txt):
            if (txt[end] in self._VOWELS) or (txt[end] == "|"):
                if end > start:
                    self.process_cons(txt[start:end], seq)
                if txt[end] == "|":
                    seq.append(self._symbol_to_id[" "])
                else:
                    seq.append(self._symbol_to_id[txt[end]])
                end += 1
                start = end
            else:
                end += 1
        if end > start:
            self.process_cons(txt[start:end], seq)
        return seq

    def sequence_to_text(self, sequence):
        """Converts a sequence of IDs back to a string"""
        result = ""
        for symbol_id in sequence:
            s = self._id_to_symbol[symbol_id]
            result += s
        return result

    def adapt_final_token(self, prev_token, tok: str, next_tok=None) -> List[str]:
        if ((tok[-1] == ".") or (tok[-1] == ",")) and (len(tok) > 1):
            return self.adapt_final_token(prev_token, tok[:-1], next_tok=next_tok) + [tok[-1:]]

        if ((tok[0] == ".") or (tok[0] == ",")) and (len(tok) > 1):
            return [tok[:1]] + self.adapt_final_token(None, tok[1:], next_tok=next_tok)

        if (tok.lower() in self.token_map) and (next_tok != "\'"):
            return self.token_map[tok.lower()]
        numer_pieces = [t for k in tok.split(",") for t in k.split(".")]
        numbers = sum([t.isnumeric() for t in numer_pieces])
        if numbers == len(numer_pieces):
            num_prefix = None if prev_token is None else self.rw_prefix(prev_token, int(numer_pieces[0][-1:]))
            if (tok.count(",") > 0) and (tok.count(".") == 1):
                if tok.rindex(".") > tok.rindex(","):
                    pieces = tok.replace(",", "").split(".")
                    num = int(pieces[0])
                    dec = int(pieces[1])
                    if dec == 1:
                        return [rw_spell_number(num_prefix, num), "n'", "igice", "kimwe"]
                    else:
                        return [rw_spell_number(num_prefix, num), "n'", "ibice", rw_spell_number("bi", dec)]
                else:
                    return [rw_spell_number(None, int(t)) for t in numer_pieces]  # Invalid number
            elif (tok.count(",") == 0) and (tok.count(".") == 1):
                pieces = tok.split(".")
                num = int(pieces[0])
                dec = int(pieces[1])
                if dec == 1:
                    return [rw_spell_number(num_prefix, num), "n'", "igice", "kimwe"]
                else:
                    return [rw_spell_number(num_prefix, num), "n'", "ibice", rw_spell_number("bi", dec)]
            elif tok.count(".") == 0:
                return [rw_spell_number(num_prefix, int(tok.replace(",", "")))]
            else:
                return [rw_spell_number(None, int(t)) for t in numer_pieces]  # Invalid number
        new_pieces = re.sub("([~!@#$%^&*()_+{}|:\"<>?`\-=\[\];\',./])', r' \1 ", tok).split()
        if new_pieces[0].isnumeric():
            num = int(new_pieces[0])
            num_prefix = None if prev_token is None else self.rw_prefix(prev_token, int(new_pieces[0][-1:]))
            return [rw_spell_number(num_prefix, num)] + [rw_spell_number(None, int(t)) if t.isnumeric() else t for t in
                                                         new_pieces[1:]]
        return [rw_spell_number(None, int(t)) if t.isnumeric() else t for t in new_pieces]


_kinyarwanda = Kinyarwanda()
tts_symbols = _kinyarwanda.tts_symbols
flexkin_socket_ready = False
uds_client = None


def init_morphokin_socket(connect_to_flexkin_socket=True,
                          sock_file="/home/nzeyi/FLEXKIN/data/run/flexkin.sock") -> uds_client:
    global flexkin_socket_ready
    global uds_client
    uds_client = None
    flexkin_socket_ready = False
    try:
        if connect_to_flexkin_socket:
            uds_client = UnixSocketClient(sock_file)
            if uds_client.connect():
                flexkin_socket_ready = True
    except Exception as ex:
        print(ex)
        flexkin_socket_ready = False
    return uds_client


def get_tokens_pos_tags_noun_prefixes(text, timing=None):
    global flexkin_socket_ready
    global uds_client
    tokens = text.split()
    if not flexkin_socket_ready:
        return [(t, "", None) for t in tokens]

    sentence = parse_text_to_morpho_sentence(uds_client, text)

    t_pos_tags = [
        f"{all_pos_tags[(t.pos_tag_id - NUM_SPECIAL_TOKENS)]['name'] if ((t.pos_tag_id - NUM_SPECIAL_TOKENS) < len(all_pos_tags)) else (t.pos_tag_id - NUM_SPECIAL_TOKENS)}"
        for t in sentence.tokens]
    t_prefix = [t.noun_class_prefix(_kinyarwanda.all_affixes) for t in sentence.tokens]
    surface_forms = [t.surface_form for t in sentence.tokens]

    if len(tokens) == len(surface_forms):
        return list(zip(tokens, t_pos_tags, t_prefix))

    alignment: needle.NeedlemanWunsch[str] = needle.NeedlemanWunsch(tokens, surface_forms)

    alignment.gap_character = "(^-^)"
    # Make the alignment
    alignment.align()

    # Get the score
    alignment.get_score()

    tokens_alignment, sf_alignments = alignment.get_aligned_sequences(core.AlignmentFormat.list)

    ret = [(t, p, np) for t, p, np in zip(tokens_alignment[:len(surface_forms)], t_pos_tags, t_prefix) if
           isinstance(t, str)]
    return ret


def spell_integer(noun, pos, numeric_token, next_token=None, prefix=None, debug=False):
    n = int(numeric_token)
    ret = rw_spell_number(('ri' if ((n % 10) == 1) else 'ka') if (
            ((noun is None) and (pos != 'UN')) or (next_token == '%')) else _kinyarwanda.rw_prefix(noun, n,
                                                                                                   prefix=prefix,
                                                                                                   pos=pos),
                          n).split()
    if debug:
        print('spell_integer:', noun, pos, numeric_token, 'next:', next_token, '==>', ret)
    return ret


def spell_decimal(noun, pos, decimal_token, next_token=None, prefix=None, debug=False):
    pieces = decimal_token.split('.')
    n = int(pieces[0])
    m = int(pieces[1])
    ret = rw_spell_number(('ri' if ((n % 10) == 1) else 'ka') if (
            ((noun is None) and (pos != 'UN')) or (next_token == '%')) else _kinyarwanda.rw_prefix(noun, n,
                                                                                                   prefix=prefix,
                                                                                                   pos=pos),
                          n).split() + ["n'", "ibice"] + rw_spell_number("bi", m).split()
    if debug:
        print('spell_decimal:', noun, pos, decimal_token, 'next:', next_token, '==>', ret)
    return ret


def is_decimal(string):
    pieces = string.split('.')
    if len(pieces) == 2:
        return pieces[0].isnumeric() and pieces[1].isnumeric()
    return False


def pre_norm_space_adjust(text: str, encoding: str = "utf-8", skip_enumerations: bool = False) -> str:
    import re
    import unicodedata
    text = text.lower()
    text = text.replace("– ", " - ")
    text = text.replace("— ", " - ")
    text = text.replace("− ", " - ")
    text = text.replace("( ", " , ")
    text = text.replace(") ", " , ")
    text = text.replace("[ ", " ")
    text = text.replace("] ", " ")
    text = text.replace("- ", " - ")
    text = text.replace(", ", " , ")
    text = text.replace("; ", " ; ")
    text = text.replace("/ ", " / ")
    tokens = []
    for token in text.split():
        if re.match(r'^\d{1,3}(,\d{3})+$', token):
            tokens.append(token.replace(",", ""))
        elif re.match(r'^\d{1,3}(\.\d{3})+$', token):
            tokens.append(token.replace(".", ""))
        else:
            tokens.append(token)
    text = ' '.join(tokens)

    if text[-1:] in ",.:;-=!?/":
        text = text[:-1] + ' ' + text[-1:]

    text = text.decode(encoding) if isinstance(text, type(b'')) else text
    text = text.replace('`', '\'')
    text = text.replace("'", "\'")
    text = text.replace("‘", "\'")
    text = text.replace("’", "\'")
    text = text.replace("‚", "\'")
    text = text.replace("‛", "\'")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")
    text = text.replace("(", " , ")
    text = text.replace(")", " , ")
    text = text.replace("[", " ")
    text = text.replace("]", " ")
    text = text.replace("-", " - ")
    text = text.replace(",", " , ")
    text = text.replace(";", " ; ")
    text = text.replace("/", " / ")
    text = text.replace('°c', 'oc')
    text = text.replace("17.17.17", "cumi na karindwi , cumi na karindwi , cumi na karindwi , ")
    text = text.replace("17-17-17", "cumi na karindwi , cumi na karindwi , cumi na karindwi , ")
    text = text.replace(u"æ", u"ae").replace(u"Æ", u"AE")
    text = text.replace(u"œ", u"oe").replace(u"Œ", u"OE")
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode("ascii").lower()
    if skip_enumerations:
        text = re.sub('^\d\.\s', '', text)

    tokens = []
    for token in text.split():
        ml = re.search(r'\d+(\.\d+)?(?=ml$)', token)
        kg = re.search(r'\d+(\.\d+)?(?=kg$)', token)
        mg = re.search(r'\d+(\.\d+)?(?=mg$)', token)
        g = re.search(r'\d+(\.\d+)?(?=g$)', token)
        cm = re.search(r'\d+(\.\d+)?(?=cm$)', token)
        l = re.search(r'\d+(\.\d+)?(?=l$)', token)
        oc = re.search(r'\d+(\.\d+)?(?=oc$)', token)
        dc = re.search(r'\d+(\.\d+)?(?=°c$)', token)
        c = re.search(r'\d+(\.\d+)?(?=c$)', token)
        m = re.search(r'\d+(\.\d+)?(?=m$)', token)
        mm = re.search(r'\d+(\.\d+)?(?=mm$)', token)
        t = re.search(r'\d+(\.\d+)?(?=t$)', token)
        ha = re.search(r'\d+(\.\d+)?(?=ha$)', token)
        if ml:
            tokens.extend(['ml', f'{ml.group()}'])
        elif kg:
            tokens.extend(['kg', f'{kg.group()}'])
        elif mg:
            tokens.extend(['mg', f'{mg.group()}'])
        elif g:
            tokens.extend(['g', f'{g.group()}'])
        elif cm:
            tokens.extend(['cm', f'{cm.group()}'])
        elif m:
            tokens.extend(['m', f'{m.group()}'])
        elif mm:
            tokens.extend(['mm', f'{mm.group()}'])
        elif l:
            tokens.extend(['l', f'{l.group()}'])
        elif t:
            tokens.extend(['toni', f'{t.group()}'])
        elif ha:
            tokens.extend(['hegitari', f'{ha.group()}'])
        elif oc:
            tokens.extend(['dogere', 'selisiyusi', f'{oc.group()}'])
        elif dc:
            tokens.extend(['dogere', 'selisiyusi', f'{dc.group()}'])
        elif c:
            tokens.extend(['dogere', 'selisiyusi', f'{c.group()}'])
        else:
            tokens.append(token)
    text = ' '.join(tokens)

    text = text.replace(" kg 1 ", " ikiro kimwe ")
    text = text.replace(" kg 1.", " ikiro 1.")

    return text


def merge_symbols_and_apostrophe(text: str, kep_time: bool = False) -> str:
    import re
    if kep_time:
        text = re.sub('([~!@#$%^&*()_+{}|"<>?`\-=\[\];\'/])', r' \1 ', text)
    else:
        text = re.sub('([~!@#$%^&*()_+{}|"<>?`\-=\[\];:\'/])', r' \1 ', text)
    out = ""
    p = ""
    for t in text.split():
        if (t == "'") and (len(p) > 0) and (len(p) < 4):
            out += t
        elif len(p) > 0:
            out += (' ' + t)
        else:
            out += t
        p = t
    return ' '.join(out.split())


def replace_hyphen(text: str) -> str:
    tokens = text.split()
    final_tokens = []
    for i, token in enumerate(tokens):
        if (token == '-') and (i > 1) and (i < (len(tokens) - 1)):
            if (tokens[i - 1].isnumeric() or is_decimal(tokens[i - 1])) and (
                    tokens[i + 1].isnumeric() or is_decimal(tokens[i + 1])):
                final_tokens.extend(['kugera', 'kuri'])
        else:
            final_tokens.append(token)
    text = ' '.join(final_tokens)
    text = text.replace("-", " ")
    text = ' '.join(text.split())
    return text


vowels = {"i", "u", "o", "a", "e"}

num_classes = {"ba", "mi"}


def hundreds(n: int) -> Union[str, None]:
    D = {1: "ijana",
         2: "magana abiri",
         3: "magana atatu",
         4: "magana ane",
         5: "magana atanu",
         6: "magana atandatu",
         7: "magana arindwi",
         8: "magana inani",
         9: "magana cyenda"}
    if n in D:
        return D[n]
    else:
        return None


def tens(n: int) -> Union[str, None]:
    D = {1: "cumi",
         2: "makumyabiri",
         3: "mirongo itatu",
         4: "mirongo ine",
         5: "mirongo itanu",
         6: "mirongo itandatu",
         7: "mirongo irindwi",
         8: "mirongo inani",
         9: "mirongo cyenda"}
    if n in D:
        return D[n]
    else:
        return None


def units(prefix: str, n: int) -> Union[str, None]:
    # print("==============units:>", prefix, n)
    D_norm = {1: "mwe",
              2: "biri",
              3: "tatu",
              4: "ne",
              5: "tanu",
              6: "tandatu",
              7: "rindwi",
              8: "umunani",
              9: "icyenda"}

    D_zi = {1: "imwe",
            2: "ebyiri",
            3: "eshatu",
            4: "enye",
            5: "eshanu",
            6: "esheshatu",
            7: "zirindwi",
            8: "umunani",
            9: "icyenda"}

    if (n == 8) or (n == 9):
        return D_norm[n]

    if prefix == "zi":
        if n in D_zi:
            return D_zi[n]
        else:
            raise Exception("Invalid number for zi")
    pair_classes = {"ba": "u",
                    "i": "u",
                    "a": "ri",
                    "bi": "ki",
                    "tu": "ka"}
    single_prefixes = {"ru", "ku", "u", "ri", "ki"}
    multiple_prefixes = {"ba", "i", "a", "bi", "zi", "ka", "tu", "bu", "ku", "ha"}
    known_prefixes = {"ru", "ku", "u", "ri", "ki", "i", "ba", "i", "a", "bi", "zi", "ka", "tu", "bu", "ku", "ha"}
    if (prefix in single_prefixes) and (n != 1):
        prefix = "ka"
    if prefix not in known_prefixes:
        prefix = "ka"
    if n in D_norm:
        suffix = D_norm[n]
        if prefix in pair_classes:
            if n == 1:
                prefix = pair_classes[prefix]
        if suffix[0:1] == "t":
            if prefix == "ka":
                prefix = "ga"
            if prefix == "ku":
                prefix = "gu"
            if prefix == "tu":
                prefix = "du"
        return prefix + suffix
    else:
        raise Exception("Invalid units number")


def hundreds_tens_units(prefix, n: int) -> str:
    if n == 10:
        return "icumi"
    h = n // 100
    t = (n % 100) // 10
    u = (n % 10)

    str = ""
    if u > 0:
        str = units(prefix, u)
        if (t > 0) or (h > 0):
            str = (" n\'" if (str[0:1] in vowels) else " na ") + str
    if t > 0:
        t_str = tens(t)
        if (h > 0):
            t_str = " na " + t_str
        str = t_str + str
    if (h > 0):
        str = hundreds(h) + str
    return str


def thousands(n: int) -> Union[str, None]:
    if n == 1:
        # return "igihumbi kimwe"
        return "igihumbi"
    elif n > 1:
        return "ibihumbi " + hundreds_tens_units("bi", n)
    else:
        return None


def millions(n: int) -> Union[str, None]:
    if n == 1:
        return "miliyoni imwe"
    elif n > 1:
        return "miliyoni " + hundreds_tens_units("zi", n)
    else:
        return None


def billions(n: int) -> Union[str, None]:
    if n == 1:
        return "miliyari imwe"
    elif n > 1:
        return "miliyari " + hundreds_tens_units("zi", n)
    else:
        return None


def trillions(n: int) -> Union[str, None]:
    if n == 1:
        return "miliyaridi imwe"
    elif n > 1:
        return "miliyaridi " + hundreds_tens_units("zi", n)
    else:
        return None


def rw_spell_number(prefix, n: int) -> Union[str, None]:
    if n == 0:
        return "zeru"

    tr = (n // 1000_000_000_000) % 1000
    bi = (n % 1000_000_000_000) // 1000_000_000
    mi = (n % 1000_000_000) // 1000_000
    th = (n % 1000_000) // 1000
    hu = (n % 1000)
    if prefix is None:
        prefix = "ri" if ((n % 10) == 1) else "ka"
    if prefix == "mi":
        prefix = "i"
    if prefix == "ma":
        prefix = "a"
    str = ""
    if hu > 0:
        hu_str = hundreds_tens_units(prefix, hu)
        if (tr > 0) or (bi > 0) or (mi > 0) or (th > 0):
            hu_str = (" n\'" if (hu_str[0:1] in vowels) else (" na " if (th == 0) else " ")) + hu_str
        str = hu_str + str
    if th > 0:
        th_str = thousands(th)
        if (tr > 0) or (bi > 0) or (mi > 0):
            th_str = " n\'" + th_str
        str = th_str + str
    if mi > 0:
        mi_str = millions(mi)
        if (tr > 0) or (bi > 0):
            mi_str = " na " + mi_str
        str = mi_str + str
    if bi > 0:
        bi_str = billions(bi)
        if (tr > 0):
            bi_str = " na " + bi_str
        str = bi_str + str
    if tr > 0:
        tr_str = trillions(tr)
        str = tr_str + str
    return str


def spell_numbers(text, timing=None, debug=False):
    text = ' '.join(text.split())
    # 10. Known token replacement
    text = ' '.join([(' '.join(_kinyarwanda.token_map[token]) if (
            (token in (_kinyarwanda.token_map)) and (token != '%')) else token) for token in text.split()])
    text = ' '.join(text.split())

    tokens_pos_tags_noun_prefixes = get_tokens_pos_tags_noun_prefixes(text, timing=timing)

    tokens = []
    prev_pos = ""
    prev_token = ""
    prev_prefix = None
    real_prev_token = ""
    real_prev_pos = ""
    for itr, (token, pos, prefix) in enumerate(tokens_pos_tags_noun_prefixes):
        prefix = 'mi' if ((token == 'iminsi') or (token == 'minsi')) else prefix
        if debug:
            print('POS/TOKEN/PREFIX:', pos, token, prefix)
        next_token = None if (itr == (len(tokens_pos_tags_noun_prefixes) - 1)) else \
            tokens_pos_tags_noun_prefixes[itr + 1][0]
        if token.isnumeric():
            tokens.extend(spell_integer(prev_token if (
                    (prev_pos == "N") or (prev_pos == "UN") or (prev_pos == "NN") or (
                    prev_token == 'selisiyusi')) else None, prev_pos, token, next_token=next_token,
                                        prefix=prev_prefix, debug=debug))
        elif is_decimal(token):
            tokens.extend(spell_decimal(prev_token if (
                    (prev_pos == "N") or (prev_pos == "UN") or (prev_pos == "NN") or (
                    prev_token == 'selisiyusi')) else None, prev_pos, token, next_token=next_token,
                                        prefix=prev_prefix, debug=debug))
        else:
            tokens.append(token)
        if (pos == 'N') or (pos == 'UN') or (pos == 'NN') or (pos == 'QA') or (pos == 'OT'):
            prev_token = token
            prev_pos = pos
            prev_prefix = prefix
        real_prev_token = token
        real_prev_pos = pos

    text = ' '.join(tokens)
    text = ' '.join(text.split())

    return text


def spell_times(text):
    tokens = []
    prev_token = ""
    for token in text.split():
        if _kinyarwanda.is_time(token) and ((prev_token == "saa") or (prev_token == "sa")):
            tokens.extend(_kinyarwanda.spell_time(token))
        else:
            tokens.append(token)
        prev_token = token
    text = ' '.join(tokens)
    text = ' '.join(text.split())
    return text


def spell_rw_phone_numbers(text):
    import re
    tokens = []
    for token in text.split():
        if re.match(_kinyarwanda._rw_phone_pattern, token):
            phones = [x for q in re.findall(_kinyarwanda._rw_phone_pattern, token) for x in q if len(x) > 8]
            for phone in phones:
                for ttii, t in enumerate(_kinyarwanda.get_phone_pieces(phone)):
                    tokens.append('guteranya' if (t == '+') else (_kinyarwanda._digits_map[int(t)]) if (
                            len(t) == 1) else rw_spell_number(None, int(t)))
                    if ttii < (len(list(phone)) - 1):
                        tokens.append(',')
        else:
            tokens.append(token)
    text = ' '.join(tokens)
    text = ' '.join(text.split())
    return text


def better_kinya_norm(raw_text, asr_text=None, encoding="utf-8", skip_enumerations=False, debug=False, timing=None):
    import re
    # 1. Pre-normalize
    text = pre_norm_space_adjust(raw_text, encoding=encoding, skip_enumerations=skip_enumerations)

    # 2. Handle numbers
    text = ' '.join([re.sub(r'(\d+(\.\d+)?)', r' \1 ', token) for token in text.split()])
    text = ' '.join(text.split())

    text = ' '.join(text.split())
    text = ' '.join([(' '.join(_kinyarwanda.token_map[token]) if (
            (token in (_kinyarwanda.token_map)) and (token != '%')) else token) for token in text.split()])
    text = ' '.join(text.split())

    # 3. Handle apostrophes
    text = merge_symbols_and_apostrophe(text, kep_time=True)

    # 4. Spell Rwanda phone numbers
    text = spell_rw_phone_numbers(text)

    # 5. Spell HH:mm:ss
    text = spell_times(text)

    # 7. Handle apostrophes again
    text = merge_symbols_and_apostrophe(text, kep_time=False)

    text = replace_hyphen(text)

    chars = re.sub(r'[^a-zA-Z0-9]', '', text)
    if any(char.isdigit() for char in chars):
        text = spell_numbers(text, timing=None, debug=debug)

    # 9. Handle apostrophes again
    text = merge_symbols_and_apostrophe(text, kep_time=False)

    text = re.sub('([~!@#$%^&*()_+{}|"<>?`\-=\[\];:.,/])', r' \1 ', text)

    text = ' '.join(text.split())
    text = ' '.join(
        [(' '.join(_kinyarwanda.token_map[token]) if (token in _kinyarwanda.token_map) else token) for token in
         text.split()])
    text = ' '.join(text.split())

    text = id_sequence_to_text(text_to_id_sequence(text))
    text = ' '.join(text.split())

    # Post replacements
    text = text.replace("kugera kuri ijana", "kugera ku ijana")
    text = text.replace("kugera kuri icumi", "kugera ku icumi")
    text = text.replace("kugera kuri icyenda", "kugera ku icyenda")
    text = text.replace("kugera kuri umunani", "kugera ku munani")
    text = text.replace("kugera kuri igihumbi", "kugera ku gihumbi")
    text = text.replace("kugera kuri ibihumbi", "kugera ku bihumbi")
    #
    text = text.replace("kugeza kuri ijana", "kugeza ku ijana")
    text = text.replace("kugeza kuri icumi", "kugeza ku icumi")
    text = text.replace("kugeza kuri umunani", "kugeza ku munani")
    text = text.replace("kugeza kuri icyenda", "kugeza ku icyenda")
    text = text.replace("kugeza kuri igihumbi", "kugeza ku gihumbi")
    text = text.replace("kugeza kuri ibihumbi", "kugeza ku bihumbi")
    text = text.replace(" ku igihumbi", " ku gihumbi")
    text = text.replace(" ku ibihumbi", " ku bihumbi")

    if asr_text is not None:
        if (text.count('kugeza ku') == 1) and (asr_text.count('kugeza ku') == 0) and (asr_text.count('kugera ku') == 1):
            text = text.replace('kugeza ku', 'kugera ku')
        if (text.count('kugera ku') == 1) and (asr_text.count('kugera ku') == 0) and (asr_text.count('kugeza ku') == 1):
            text = text.replace('kugera ku', 'kugeza ku')

        if (text.count(' ibiro ') == 1) and (text.count(' ikiro ') == 0) and (text.count(' kilogarama ') == 0) and (
                text.count(' kirogarama ') == 0) and (asr_text.count(' ibiro ') == 0) and (
                asr_text.count(' kilogarama ') == 1):
            text = text.replace(' ibiro ', ' kilogarama ')
        if (text.count(' ku biro ') == 1) and (text.count(' ku kiro ') == 0) and (
                text.count(' kuri kilogarama ') == 0) and (text.count(' kuri kirogarama ') == 0) and (
                asr_text.count(' ku biro ') == 0) and (asr_text.count(' kuri kilogarama ') == 1):
            text = text.replace(' ku biro ', ' kuri kilogarama ')

        if (text.count(' ibiro ') == 1) and (text.count(' ikiro ') == 0) and (text.count(' kilogarama ') == 0) and (
                text.count(' kirogarama ') == 0) and (asr_text.count(' ibiro ') == 0) and (
                asr_text.count(' kirogarama ') == 1):
            text = text.replace(' ibiro ', ' kirogarama ')
        if (text.count(' ku biro ') == 1) and (text.count(' ku kiro ') == 0) and (
                text.count(' kuri kilogarama ') == 0) and (text.count(' kuri kirogarama ') == 0) and (
                asr_text.count(' ku biro ') == 0) and (asr_text.count(' kuri kirogarama ') == 1):
            text = text.replace(' ku biro ', ' kuri kirogarama ')

        if (text.count(' ikiro ') == 1) and (text.count(' ibiro ') == 0) and (text.count(' kilogarama ') == 0) and (
                text.count(' kirogarama ') == 0) and (asr_text.count(' ikiro ') == 0) and (
                asr_text.count(' kilogarama ') == 1):
            text = text.replace(' ikiro ', ' kilogarama ')
        if (text.count(' ku kiro ') == 1) and (text.count(' ku biro ') == 0) and (
                text.count(' kuri kilogarama ') == 0) and (text.count(' kuri kirogarama ') == 0) and (
                asr_text.count(' ku kiro ') == 0) and (asr_text.count(' kuri kilogarama ') == 1):
            text = text.replace(' ku kiro ', ' kuri kilogarama ')

        if (text.count(' ikiro ') == 1) and (text.count(' ibiro ') == 0) and (text.count(' kilogarama ') == 0) and (
                text.count(' kirogarama ') == 0) and (asr_text.count(' ikiro ') == 0) and (
                asr_text.count(' kirogarama ') == 1):
            text = text.replace(' ikiro ', ' kirogarama ')
        if (text.count(' ku kiro ') == 1) and (text.count(' ku biro ') == 0) and (
                text.count(' kuri kilogarama ') == 0) and (text.count(' kuri kirogarama ') == 0) and (
                asr_text.count(' ku kiro ') == 0) and (asr_text.count(' kuri kirogarama ') == 1):
            text = text.replace(' ku kiro ', ' kuri kirogarama ')

        if re.match(r'\.5[^0-9]', raw_text):
            if (text.count('bice bitanu ') == 1) and (text.count('gice ') == 0) and (
                    asr_text.count('bice bitanu ') == 0) and (
                    (asr_text.count('gice ') == 1) or (asr_text.count(' nigice ') == 1)):
                text = text.replace("n' ibice bitanu ", "n' igice ")

    return text


def text_to_tts_sequence(text, norm=False):
    if norm:
        text = norm_text(text)
    return _kinyarwanda.txt2seq(text)


def tts_sequence_to_text(seq):
    return _kinyarwanda.sequence_to_text(seq)


def norm_text(text, asr_text=None, encoding="utf-8", skip_enumerations=False, timing=None, debug=False) -> str:
    return better_kinya_norm(text, asr_text=asr_text, encoding=encoding, skip_enumerations=skip_enumerations,
                             timing=timing, debug=debug)
