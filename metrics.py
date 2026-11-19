import re


def normalize(text):
    text = text.lower()
    text = text.replace("-", " ")
    # keep letters, digits, apostrophes and spaces
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    text = text.replace(" '", " ").replace("' ", " ")
    return " ".join(text.split())


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
