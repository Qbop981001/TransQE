import sys
RAW_FILE = sys.argv[1]
OUT_SRC=sys.argv[2]
OUT_TGT=sys.argv[3]


def read_file(filename):
    data = []
    s_o, t_o = [], []
    with open(filename,encoding='utf-8') as f:
        data = f.readlines()

    for i in range(len(data)):
        if data[i][:4] == '<doc':
        # if '<doc origlang' in data[i]:
            # print(data[i])
            if 'origlang="en"' in data[i]:
                i+=2
                while data[i][:4] != '</p>':
                # while '</src>' not in data[i]:
                    s_o.append(data[i])
                    i+=1
            elif 'origlang="de"' in data[i]:
                i+=2
                while data[i][:4] != '</p>':
                    t_o.append(data[i])
                    i+=1
    print(s_o)
    for i in range(len(s_o)):
        tmp=s_o[i]
        index_s = tmp.find('">') + 2
        index_end = tmp.find('</')
        tmp = tmp[index_s:index_end]
        s_o[i] = tmp
    for i in range(len(t_o)):
        tmp=t_o[i]
        index_s = tmp.find('">') + 2
        index_end = tmp.find('</')
        tmp = tmp[index_s:index_end]
        t_o[i] = tmp
    return s_o, t_o


def write_file(filename,data):
    if not data:
        return
    with open(filename,mode='w',encoding='utf-8') as f:
        for line in data:
            f.write(line+'\n')


def read_21_file(filename):

    src_text, ref_text = [], []
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    root = tree.getroot()
    docs = root.findall('doc')
    for doc in docs:
        src = doc.findall('src')[0].findall('p')
        ref = doc.findall('ref')[0].findall('p')
        assert (len(src)==len(ref))
        for pa,pb in zip(src,ref):
            pa=pa.findall('seg')
            pb=pb.findall('seg')
            print(len(pa),len(pb))
            for a,b in zip(pa,pb):
                src_text.append(a.text)
                ref_text.append(b.text)

    return  src_text, ref_text
# s_o, t_o = read_file(RAW_FILE)
srcs, tgts = read_21_file(RAW_FILE)


# print(len(s_o))
# print(len(t_o))
# write_file(OUT_SRC,s_o)
# write_file(OUT_TGT,t_o)




# write_file('/workspace/translationese/data/test21.orig.src',srcs)
# write_file('/workspace/translationese/data/test21.orig',tgts)


