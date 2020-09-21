# エトセトラ．


from observations import ptb


def test_chr_to_ascii():
    """ 文字とアスキーコードの間の変換．
    """
    assert ord('a') == 97
    assert ord('b') == 98
    assert ord('c') == 99
    assert chr(97) == 'a'
    assert chr(98) == 'b'
    assert chr(99) == 'c'


def test_unique_chars_in_raw_ptb():
    """ Penn Treebank 内のユニークな文字の種類数の確認．
    """
    # Penn Treebank を取得する．
    # 訓練用，テスト用，検証用の文字列が取得される．
    x_train, x_test, x_valid = ptb("./data")  # 初回はダウンロードが走る．
    s = set([c for c in x_train])
    s = s | set([c for c in x_test])
    s = s | set([c for c in x_valid])

    assert len(s) == 49  # 49文字

    s = list(s)
    s = [ord(c) for c in s]
    s.sort()
    assert s[-1] == 122  # 122番までしか使われていない．
