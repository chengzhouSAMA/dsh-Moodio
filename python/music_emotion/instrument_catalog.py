# -*- coding: utf-8 -*-
"""用户指定的 21 种目标乐器；英文声学描述供音频文本模型对齐。"""

INSTRUMENTS = [
    ("guzheng", "古筝", "plucked_string", "Chinese guzheng plucked zither with many strings and sweeping glissandos"),
    ("erhu", "二胡", "bowed_string", "Chinese erhu two-string bowed fiddle with expressive sliding notes"),
    ("pipa", "琵琶", "plucked_string", "Chinese pipa plucked lute with bright tremolo and fast plucked attacks"),
    ("dizi", "笛子", "flute", "Chinese dizi bamboo transverse flute with a bright buzzing membrane tone"),
    ("hulusi", "葫芦丝", "free_reed", "Chinese hulusi gourd flute with a mellow free-reed tone and drone"),
    ("suona", "唢呐", "double_reed", "Chinese suona double-reed horn with a loud piercing nasal tone"),
    ("yangqin", "扬琴", "struck_string", "Chinese yangqin hammered dulcimer with ringing strings struck by small hammers"),
    ("sheng", "笙", "free_reed", "Chinese sheng bamboo mouth organ with sustained free-reed chords"),
    ("ocarina", "陶笛", "flute", "ocarina vessel flute with a round pure breathy tone"),
    ("guqin", "古琴", "plucked_string", "Chinese guqin seven-string zither with soft plucks, harmonics and sliding tones"),
    ("piano", "钢琴", "struck_string", "acoustic piano playing struck string notes and chords"),
    ("guitar", "吉他", "plucked_string", "guitar playing plucked strings and strummed chords"),
    ("violin", "小提琴", "bowed_string", "violin playing high bowed string melodies with vibrato"),
    ("saxophone", "萨克斯", "single_reed", "saxophone playing expressive breathy reed melodies"),
    ("drum_kit", "架子鼓", "percussion", "drum kit playing kick drum, snare, toms, hi-hat and cymbals"),
    ("cello", "大提琴", "bowed_string", "cello playing low warm bowed string melodies"),
    ("flute", "长笛", "flute", "Western concert flute playing clear airy transverse flute melodies"),
    ("bass", "贝斯", "plucked_string", "electric bass guitar playing low plucked rhythmic bass notes"),
    ("ukulele", "尤克里里", "plucked_string", "ukulele playing small bright nylon-string strummed chords"),
    ("harmonica", "口琴", "free_reed", "harmonica mouth organ playing bending free-reed notes"),
    ("accordion", "手风琴", "free_reed", "accordion playing sustained reed chords and melodies with bellows"),
]

OTHER_SOUNDS = [
    "A recording of people speaking.",
    "A recording of a person singing without instruments.",
    "A recording of electronic synthesizer sounds.",
    "A recording of environmental noise without music.",
    "A recording of trumpet and brass instruments.",
    "A recording of a clarinet playing music.",
]

AST_DIRECT_LABELS = {
    "piano": ["Piano", "Electric piano"], "guitar": ["Guitar", "Acoustic guitar", "Electric guitar"],
    "violin": ["Violin, fiddle"], "saxophone": ["Saxophone"], "drum_kit": ["Drum kit"],
    "cello": ["Cello"], "flute": ["Flute"], "bass": ["Bass guitar"], "ukulele": ["Ukulele"],
    "harmonica": ["Harmonica"], "accordion": ["Accordion"],
}
AST_FAMILY_LABELS = {
    "guzheng": ["Zither"], "guqin": ["Zither"], "erhu": ["Bowed string instrument"],
    "pipa": ["Plucked string instrument"], "dizi": ["Flute"], "ocarina": ["Flute"],
    "hulusi": ["Wind instrument, woodwind instrument"], "suona": ["Wind instrument, woodwind instrument"],
    "sheng": ["Wind instrument, woodwind instrument"],
}


def catalog():
    return [{"id": identifier, "name": name, "family": family,
             "coverage": "zero_shot_target_not_accuracy_guarantee",
             "crosscheck_scope": "direct_label" if identifier in AST_DIRECT_LABELS else
                                 "family_only" if identifier in AST_FAMILY_LABELS else "zero_shot_only"}
            for identifier, name, family, _ in INSTRUMENTS]


def prompts():
    return [text for _, _, _, description in INSTRUMENTS
            for text in (f"The sound of {description}.", f"A music recording featuring {description}.")]
