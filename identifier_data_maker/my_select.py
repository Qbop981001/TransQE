import sys
lang=sys.argv[1]
threshold=int(sys.argv[2])

def read_file(filename):
    res=[]
    count=0
    with open(filename,'r') as f:
        lines=f.readlines()
        for line in lines:
            if len(line) < threshold:
                res.append(line.strip())
                count+=1
            if count == 1000000: break

    return res


def write_file(filename,out):
    with open(filename,'w') as f:
        for item in out:
            f.write(item+'\n')



in_file=f"/home/njuciairs/qbp/jd/exp_data/mono/news.2021.{lang}.shuffled.deduped"
out_file=f"/home/njuciairs/qbp/jd/exp_data/mono/news.2021.mono.{lang}"
data = read_file(in_file)
write_file(out_file, data)
