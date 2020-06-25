#!/usr/bin/env python3


def main():
    cpfx = 1
    ipfx = 1000000001

    ihash = {}
    with open("unique_sid.txt", "w") as fout:
        with open("original/unique_sid.txt") as fin:
            for line in fin:
                i = int(line)
                assert i > 0
                assert i not in ihash
                ihash[i] = ipfx + len(ihash)
                print(ihash[i], file=fout)

    chash = {}
    with open("item_cate.csv", "w") as fout:
        with open("original/item_cate.csv") as fin:
            for line in fin:
                i, c = line.split()
                i, c = int(i), int(c)
                if i not in ihash:
                    print('!', i)
                    continue
                assert c > 0
                if c not in chash:
                    chash[c] = cpfx + len(chash)
                print('%d,%d' % (ihash[i], chash[c]), file=fout)

    with open("item_detail.csv", "w") as fout:
        with open("original/mini_item_feature_v2") as fin:
            for line in fin:
                cols = [col.strip() for col in line.split(',')]
                item_id, cate_level, cate_id, cate_name = cols[:4]
                cate_id_at_lv = cols[4:9]
                cate_name_at_lv = cols[9:14]
                item_title = ' '.join(cols[14:-1])  # title may contains ','
                pict_url = cols[-1]
                item_id = int(item_id)
                if item_id not in ihash:
                    print('@', item_id)
                    continue
                cate_leval, cate_id = int(cate_level) - 1, int(cate_id)
                cate_id_at_lv = [(-1 if c == '' else int(c))
                                 for c in cate_id_at_lv]
                assert cate_id_at_lv[cate_leval] == cate_id
                assert cate_name_at_lv[cate_leval] == cate_name
                if cate_name == '其它':
                    assert cate_leval > 0
                    cate_name = cate_name_at_lv[cate_leval - 1]
                    print('其它->', cate_name)
                pict_url = 'https://img.alicdn.com/imgextra/' + pict_url
                item_id = ihash[item_id]
                assert ',' not in cate_name
                assert ',' not in item_title
                assert ',' not in pict_url
                print('%d,%s,%s,%s' % (
                    item_id, cate_name, item_title, pict_url), file=fout)


if __name__ == '__main__':
    main()
