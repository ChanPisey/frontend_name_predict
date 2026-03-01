"""
Khmer Character Cluster (KCC) Helper Functions
"""

# Khmer character definitions
KHCONST = list(u'កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ')
KHVOWEL = list(u'឴឵ាិីឹឺុូួើឿៀេែៃោៅ\u17c6\u17c7\u17c8')
KHSUB = list(u'្')
KHDIAC = list(u"\u17c9\u17ca\u17cb\u17cc\u17cd\u17ce\u17cf\u17d0")
KHSYM = list('៕។៛ៗ៚៙៘,.? ')
KHNUMBER = list(u'០១២៣៤៥៦៧៨៩0123456789')
KHLUNAR = list('᧠᧡᧢᧣᧤᧥᧦᧧᧨᧩᧪᧫᧬᧭᧮᧯᧰᧱᧲᧳᧴᧵᧶᧷᧸᧹᧺᧻᧼᧽᧾᧿')
EN = set(u'abcdefghijklmnopqrstuvwxyz0123456789')


def is_khmer_char(ch):
    """Check if a character is Khmer"""
    if (ch >= '\u1780') and (ch <= '\u17ff'):
        return True
    if ch in KHSYM or ch in KHLUNAR:
        return True
    return False


def is_start_of_kcc(ch):
    """Check if a character can start a new KCC"""
    if is_khmer_char(ch):
        if ch in KHCONST or ch in KHSYM or ch in KHNUMBER or ch in KHLUNAR:
            return True
        return False
    return True


def seg_kcc(str_sentence):
    """Segment a Khmer sentence into KCCs (Khmer Character Clusters)"""
    segs = []
    cur = ""
    sentence = str_sentence

    for word in sentence.split('\u200b'):
        for i, c in enumerate(word):
            cur += c
            nextchar = word[i+1] if (i+1 < len(word)) else ""

            if not is_khmer_char(c) and nextchar != " " and nextchar != "" and not is_khmer_char(nextchar):
                continue

            if c in KHNUMBER and nextchar in KHNUMBER:
                continue

            if not is_khmer_char(c) or nextchar == " " or nextchar == "":
                segs.append(cur)
                cur = ""
            elif is_start_of_kcc(nextchar) and not (c in KHSUB):
                segs.append(cur)
                cur = ""

    return segs


def cleanup_str(text):
    """Clean up Khmer text for processing"""
    text = text.strip('\u200b').strip()
    text = text.replace("  ", " ")
    text = text.replace(" ", "\u200b")
    text = text.replace("\u200b\u200b", '\u200b')
    text = text.replace(u"\u2028", "")
    text = text.replace(u"\u200a", "")
    text = text.strip().replace('\n', '').replace('  ', ' ')
    return text
