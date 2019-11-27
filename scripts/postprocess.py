import json
import os
import tarfile
from datetime import datetime
from glob import glob
import collections
import argparse


def make_dirs(dirname):
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def read_text(filename, encoding="UTF-8"):
    with open(filename, "r", encoding=encoding) as f:
        return f.read()


def read_lines(filename, encoding="UTF-8"):
    with open(filename, "r", encoding=encoding) as f:
        for line in f:
            yield line.rstrip("\r\n\v")


def write_lines(lines, filename, linesep="\n", encoding="UTF-8"):
    make_dirs(os.path.dirname(filename))
    with open(filename, "w", encoding=encoding) as f:
        for line in lines:
            f.write(line)
            f.write(linesep)


def read_json(filename, encoding="UTF-8"):
    return json.loads(read_text(filename, encoding=encoding))


parser = argparse.ArgumentParser()
parser.add_argument('--corpusdir', type=str, required=True, help='--corpusdir')
parser.add_argument('--indir', type=str, required=True, help='--indir')
parser.add_argument('--outdir', type=str, required=True, help='--outdir')
args = parser.parse_args()

corpus_dir = getattr(args, 'corpusdir')
input_dir = getattr(args, 'indir')
outdir = getattr(args, 'outdir')

output_dir = outdir + 'ev-ann-out'
if not os.path.exists(output_dir):
    make_dirs(output_dir)
else:
    os.system('rm ' + output_dir + '/*.a2')

assert (
    len(glob(os.path.join(output_dir, "**/*"), recursive=True)) == 0
), "The folder `{}` must be empty!".format(output_dir)

count = 0


for cur_fn in glob(os.path.join(input_dir, "**/*.a2"), recursive=True):

    offset_mapping = read_json(
        os.path.join(corpus_dir, os.path.basename(cur_fn).replace(".a2", ".inv.map"))
    )
    reference = read_text(
        os.path.join(corpus_dir, os.path.basename(cur_fn).replace(".a2", ".txt.ori"))
    )

    gold_entities = offset_mapping["entities"]

    processed_lines = []

    valid_eid_list = []
    saved_eid_list = []
    saved_edata_list = []
    dup_eid_list = []
    for line in read_lines(cur_fn):
        if line.startswith("E"):
            line_sp = line.split("\t")
            eid = line_sp[0]
            e_data = line_sp[1].split(" ")
            e_data = collections.Counter(e_data)
           
            if e_data not in saved_edata_list:
                saved_edata_list.append(e_data)
                valid_eid_list.append(eid)
            else:
                dup_eid_list.append(eid)

    for line in read_lines(cur_fn):
        if line.startswith("T"):
            _id, _attrs, _ = line.split("\t")
            _type, *_offsets = _attrs.split()
            _start, _end = map(lambda x: offset_mapping[x], _offsets)

            # assert _id in gold_entities and (_start, _end) == gold_entities[_id]

            processed_lines.append(
                "{}\t{} {} {}\t{}".format(
                    _id, _type, _start, _end, reference[_start:_end]
                )
            )
        elif line.startswith("E"):
            line_sp = line.split("\t")
            eid = line_sp[0]
            e_data = line[len(eid):].strip()

            if eid in valid_eid_list:
                is_valid = True
                for eid2 in dup_eid_list:
                    if eid2 in e_data:
                        is_valid = False
                        break

                if is_valid:
                    saved_eid_list.append(eid)
                    processed_lines.append(line)
        elif line.startswith("M"):
            line_sp = line.split("\t")
            mid = line_sp[0]
            eid = line_sp[1].split(" ")[1]
            if eid in saved_eid_list:
                processed_lines.append(line)

    write_lines(processed_lines, os.path.join(output_dir, os.path.basename(cur_fn)))

    count += 1
    print(os.path.basename(cur_fn), "Done")

print("Processed {} files".format(count))

# write empty predicted files
print('EMPTY FILES:')
for ref_fn in glob(os.path.join(corpus_dir, "**/*.a2"), recursive=True):

    pred_fn = os.path.join(input_dir, os.path.basename(ref_fn))

    if os.path.isfile(pred_fn):
        continue
    else:
        print(ref_fn)

        # write empty file
        write_lines([], os.path.join(output_dir, os.path.basename(ref_fn)))

if count > 0:
    tgz_fn = "{}.tar.gz".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    outfile_path = outdir + tgz_fn
    with tarfile.open(outfile_path, "w:gz") as f:
        for fn in glob(os.path.join(output_dir, "*.a2")):
            f.add(fn, arcname=os.path.basename(fn))
        print("Please submit this file: {}".format(outfile_path))
