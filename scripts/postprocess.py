import json
import os
import tarfile
from datetime import datetime
from glob import glob
import collections
import sys
import shutil


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


def retrieve_offset_a2(refdir, preddir, outdir, corpus_name, dev_test):
    # output dir
    output_a2_dir = outdir + 'ev-orig-a2'
    zip_dir = outdir + 'online-eval'

    # create output dirs
    if not os.path.exists(output_a2_dir):
        make_dirs(output_a2_dir)
    else:
        os.system('rm -rf ' + output_a2_dir + '/*')

    # assert (
    #         len(glob(os.path.join(output_a2_dir, "**/*"), recursive=True)) == 0
    # ), "The folder `{}` must be empty!".format(output_a2_dir)

    if not os.path.exists(zip_dir):
        make_dirs(zip_dir)

    count = 0

    for cur_fn in glob(os.path.join(preddir, "**/*.a2"), recursive=True):

        offset_mapping = read_json(
            os.path.join(refdir, os.path.basename(cur_fn).replace(".a2", ".inv.map"))
        )
        reference = read_text(
            os.path.join(refdir, os.path.basename(cur_fn).replace(".a2", ".txt.ori"))
        )

        # gold_entities = offset_mapping["entities"]

        processed_lines = []

        # for brat
        processed_ann_lines = []

        valid_eid_list = []
        saved_eid_list = []
        saved_edata_list = []
        invalid_eid_list = []

        # read a1
        for line in read_lines(os.path.join(refdir, os.path.basename(cur_fn).replace(".a2", ".a1"))):
            if line.startswith('T'):
                _id, _attrs, _ = line.split("\t")
                _type, *_offsets = _attrs.split()
                _start, _end = map(lambda x: offset_mapping[x], _offsets)

                # assert _id in gold_entities and (_start, _end) == gold_entities[_id]

                processed_ann_lines.append("{}\t{} {} {}\t{}".format(
                    _id, _type, _start, _end, reference[_start:_end]
                ))

        for line in read_lines(cur_fn):
            if line.startswith("E"):
                line_sp = line.split("\t")
                eid = line_sp[0]
                e_data = line_sp[1].split(" ")
                e_data2 = collections.Counter(e_data)

                valid = True

                if e_data2 not in saved_edata_list:
                    for arg in e_data:
                        argeid = arg.split(':')[1]

                        # argument is in the duplicated event list
                        if argeid in invalid_eid_list:
                            valid = False
                            break

                # event is duplicated
                else:
                    valid = False

                # for ge13: no Cause but CSite
                if 'CSite' in line:
                    if not 'Cause' in line:
                        valid = False

                if valid:
                    saved_edata_list.append(e_data2)
                    valid_eid_list.append(eid)
                else:
                    invalid_eid_list.append(eid)

        for line in read_lines(cur_fn):
            if line.startswith("T"):
                _id, _attrs, _ = line.split("\t")
                _type, *_offsets = _attrs.split()
                _start, _end = map(lambda x: offset_mapping[x], _offsets)

                # assert _id in gold_entities and (_start, _end) == gold_entities[_id]

                ent_line = "{}\t{} {} {}\t{}".format(
                    _id, _type, _start, _end, reference[_start:_end]
                )

                # for a2
                processed_lines.append(ent_line)

                # for ann
                processed_ann_lines.append(ent_line)

            elif line.startswith("E"):
                line_sp = line.split("\t")
                eid = line_sp[0]

                if eid in valid_eid_list:
                    saved_eid_list.append(eid)
                    processed_lines.append(line)
                    processed_ann_lines.append(line)

            elif line.startswith("M"):
                line_sp = line.split("\t")
                mid = line_sp[0]
                eid = line_sp[1].split(" ")[1]
                if eid in saved_eid_list:
                    processed_lines.append(line)
                    processed_ann_lines.append(line)

        # write a2
        write_lines(processed_lines, os.path.join(output_a2_dir, os.path.basename(cur_fn)))

        count += 1
        print(os.path.basename(cur_fn), "Done")

    print("Processed {} files".format(count))

    # write empty predicted files
    for ref_fn in glob(os.path.join(refdir, "**/*.a2"), recursive=True):

        pred_fn = os.path.join(preddir, os.path.basename(ref_fn))

        if os.path.isfile(pred_fn):
            continue
        else:
            print('empty file: ', ref_fn)

            # write empty file
            write_lines([], os.path.join(output_a2_dir, os.path.basename(ref_fn)))

    # create zip format for online evaluation
    if count > 0:

        # zip file name
        tgz_fn = "{}.tar.gz".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        zip_file_name = ''.join([corpus_name, '-', dev_test, '-', tgz_fn])

        # zip file path
        outfile_path = os.path.join(zip_dir, zip_file_name)
        with tarfile.open(outfile_path, "w:gz") as f:
            for fn in glob(os.path.join(output_a2_dir, "*.a2")):
                f.add(fn, arcname=os.path.basename(fn))
            if 'dev' in dev_test or 'test' in dev_test:
                print("Please submit this file: {}".format(outfile_path))


def retrieve_offset_ann(refdir, preddir, outdir, corpus_name):
    # output dir
    output_ann_dir = ''.join([outdir, corpus_name, '-', 'brat'])

    # create output dirs

    if not os.path.exists(output_ann_dir):
        make_dirs(output_ann_dir)
    else:
        os.system('rm -rf ' + output_ann_dir + '/*')

    # assert (
    #         len(glob(os.path.join(output_ann_dir, "**/*"), recursive=True)) == 0
    # ), "The folder `{}` must be empty!".format(output_ann_dir)

    count = 0

    for cur_fn in glob(os.path.join(preddir, "**/*.a2"), recursive=True):

        offset_mapping = read_json(
            os.path.join(refdir, os.path.basename(cur_fn).replace(".a2", ".inv.map"))
        )
        reference = read_text(
            os.path.join(refdir, os.path.basename(cur_fn).replace(".a2", ".txt.ori"))
        )

        # gold_entities = offset_mapping["entities"]

        processed_lines = []

        # for brat
        processed_ann_lines = []

        valid_eid_list = []
        saved_eid_list = []
        saved_edata_list = []
        invalid_eid_list = []

        # read a1
        for line in read_lines(os.path.join(refdir, os.path.basename(cur_fn).replace(".a2", ".a1"))):
            if line.startswith('T'):
                _id, _attrs, _ = line.split("\t")
                _type, *_offsets = _attrs.split()
                _start, _end = map(lambda x: offset_mapping[x], _offsets)

                # assert _id in gold_entities and (_start, _end) == gold_entities[_id]

                processed_ann_lines.append("{}\t{} {} {}\t{}".format(
                    _id, _type, _start, _end, reference[_start:_end]
                ))

        for line in read_lines(cur_fn):
            if line.startswith("E"):
                line_sp = line.split("\t")
                eid = line_sp[0]
                e_data = line_sp[1].split(" ")
                e_data2 = collections.Counter(e_data)

                valid = True

                if e_data2 not in saved_edata_list:
                    for arg in e_data:
                        argeid = arg.split(':')[1]

                        # argument is in the duplicated event list
                        if argeid in invalid_eid_list:
                            valid = False
                            break

                # event is duplicated
                else:
                    valid = False

                # for ge13: no Cause but CSite
                if 'CSite' in line:
                    if not 'Cause' in line:
                        valid = False

                if valid:
                    saved_edata_list.append(e_data2)
                    valid_eid_list.append(eid)
                else:
                    invalid_eid_list.append(eid)

        for line in read_lines(cur_fn):
            if line.startswith("T"):
                _id, _attrs, _ = line.split("\t")
                _type, *_offsets = _attrs.split()
                _start, _end = map(lambda x: offset_mapping[x], _offsets)

                # assert _id in gold_entities and (_start, _end) == gold_entities[_id]

                ent_line = "{}\t{} {} {}\t{}".format(
                    _id, _type, _start, _end, reference[_start:_end]
                )

                # for a2
                processed_lines.append(ent_line)

                # for ann
                processed_ann_lines.append(ent_line)

            elif line.startswith("E"):
                line_sp = line.split("\t")
                eid = line_sp[0]

                if eid in valid_eid_list:
                    saved_eid_list.append(eid)
                    processed_lines.append(line)
                    processed_ann_lines.append(line)

            elif line.startswith("M"):
                line_sp = line.split("\t")
                mid = line_sp[0]
                eid = line_sp[1].split(" ")[1]
                if eid in saved_eid_list:
                    processed_lines.append(line)
                    processed_ann_lines.append(line)

        # write ann for brat
        write_lines(processed_ann_lines, os.path.join(output_ann_dir, os.path.basename(cur_fn.replace(".a2", ".ann"))))

        # write txt
        txt_fn = os.path.basename(cur_fn).replace(".a2", ".txt.ori")
        shutil.copy(os.path.join(refdir, txt_fn), os.path.join(output_ann_dir, txt_fn.replace(".txt.ori", ".txt")))

        count += 1
        print(os.path.basename(cur_fn), "Done")

    print("Processed {} files".format(count))

    # write empty predicted files
    for ref_fn in glob(os.path.join(refdir, "**/*.a2"), recursive=True):

        pred_fn = os.path.join(preddir, os.path.basename(ref_fn))

        if os.path.isfile(pred_fn):
            continue
        else:
            print('empty file: ', ref_fn)

            # write empty file
            write_lines([], os.path.join(output_ann_dir, os.path.basename(ref_fn.replace(".a2", ".ann"))))

            # write txt
            txt_fn = os.path.basename(ref_fn).replace(".a2", ".txt.ori")
            shutil.copy(os.path.join(refdir, txt_fn), os.path.join(output_ann_dir, txt_fn.replace(".txt.ori", ".txt")))


if __name__ == '__main__':

    # debug
    # corpus_name = 'cg'
    # outdir = '../experiments/cg/predict-gold-dev/ev-last/'
    # preddir = '../experiments/cg/predict-gold-dev/ev-last/ev-tok-a2/'
    # refdir = '../data/corpora/cg/dev/'
    # dev_test = 'dev'

    # a2 files
    if len(sys.argv) == 6:
        retrieve_offset_a2(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

    # ann files
    elif len(sys.argv) == 5:
        retrieve_offset_ann(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
