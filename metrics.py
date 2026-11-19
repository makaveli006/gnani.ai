import re

# Prisma writes "she's" as "she is". Both are correct, so expand
# contractions on both sides before comparing.
CONTRACTIONS = {
    "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
    "he's": "he is", "she's": "she is", "it's": "it is", "that's": "that is",
    "what's": "what is", "there's": "there is", "let's": "let us",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "couldn't": "could not", "wouldn't": "would not",
    "i've": "i have", "you've": "you have", "we've": "we have", "i'll": "i will",
    "you'll": "you will", "i'd": "i would",
}


def normalize(text):
    text = text.lower()
    text = text.replace("-", " ").replace("’", "'")
    # keep letters, digits, apostrophes and spaces
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    words = [CONTRACTIONS.get(w.strip("'"), w.strip("'")) for w in text.split()]
    return " ".join(" ".join(words).split())


def edit_ops(ref, hyp):
    """Levenshtein alignment between two token lists.
    Returns (substitutions, deletions, insertions)."""
    n, m = len(ref), len(hyp)

    # d[i][j] = (cost, S, D, I) for ref[:i] vs hyp[:j]
    d = [[None] * (m + 1) for _ in range(n + 1)]
    d[0][0] = (0, 0, 0, 0)
    for i in range(1, n + 1):
        d[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        d[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
                continue
            c, s, dl, ins = d[i - 1][j - 1]
            sub = (c + 1, s + 1, dl, ins)
            c, s, dl, ins = d[i - 1][j]
            dele = (c + 1, s, dl + 1, ins)
            c, s, dl, ins = d[i][j - 1]
            inse = (c + 1, s, dl, ins + 1)
            d[i][j] = min(sub, dele, inse)

    _, s, dl, ins = d[n][m]
    return s, dl, ins


def wer(ref, hyp):
    # WER = (S + D + I) / number of words in the reference
    r = normalize(ref).split()
    h = normalize(hyp).split()
    s, d, i = edit_ops(r, h)
    return (s + d + i) / max(len(r), 1)


def cer(ref, hyp):
    # same idea on characters, spaces ignored
    r = list(normalize(ref).replace(" ", ""))
    h = list(normalize(hyp).replace(" ", ""))
    s, d, i = edit_ops(r, h)
    return (s + d + i) / max(len(r), 1)


def error_counts(ref, hyp):
    """Counts used for corpus-level WER/CER:
    total errors / total reference words (or characters)."""
    r = normalize(ref).split()
    h = normalize(hyp).split()
    s, d, i = edit_ops(r, h)
    rc = list("".join(r))
    hc = list("".join(h))
    cs, cd, ci = edit_ops(rc, hc)
    return {
        "ref_words": len(r), "sub": s, "del": d, "ins": i,
        "ref_chars": len(rc), "char_errors": cs + cd + ci,
    }
