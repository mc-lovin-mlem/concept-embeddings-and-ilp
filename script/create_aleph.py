import argparse
import os

parser = argparse.ArgumentParser(description="Create the prolog scripts from annotation .csv files.")
parser.add_argument('--root', type=str,
                    default=os.path.join("experiments", "ilp"),
                    help="The root directory in which to find the annotation files and where to put the "
                         "aleph .pl files (see also --exp_name).")
parser.add_argument('--exp_name', type=str,
                    default=None,
                    help='If --root is not given but --exp_name, root defaults to experiments/ilp/exp_name.'
                         'exp_name is usually the folder name of the experiment root folder '
                         'for the corresponding concept analysis.')
PROJECT_ROOT = "."  # assume the script is called from within project root
args = parser.parse_args()
root = args.root
if root is None:
    root = os.path.join(PROJECT_ROOT, "experiments", "ilp")
    if args.exp_name is not None:
        dest_root = os.path.join(root, args.exp_name)


def get_constellations(obj_coords, ref_coords):
    output = []
    obj_avg_row = (obj_coords[0] + obj_coords[1]) // 2
    obj_avg_col = (obj_coords[2] + obj_coords[3]) // 2
    ref_avg_row = (ref_coords[0] + ref_coords[1]) // 2
    ref_avg_col = (ref_coords[2] + ref_coords[3]) // 2

    row_offset = obj_avg_row - ref_avg_row
    col_offset = obj_avg_col - ref_avg_col

    if obj_avg_row < ref_avg_row:
        if abs(col_offset) * .5 < abs(row_offset):
            output.append("top_of")
    if obj_avg_col < ref_avg_col:
        if abs(row_offset) * .5 < abs(col_offset):
            output.append("left_of")

    return output


anno_file_p = open(os.path.join(root, "annotations_positive.csv"), 'r')
anno_lines_p = anno_file_p.read().splitlines()

anno_file_n = open(os.path.join(root, "annotations_negative.csv"), 'r')
anno_lines_n = anno_file_n.read().splitlines()

N_POS = len(anno_lines_p) // 9
N_NEG = len(anno_lines_n) // 9
print("POS:", N_POS, "NEG:", N_NEG)
pos_list = []
neg_list = []
for i in range(N_POS):
    organ_dict = {}
    for o in range(4):
        name = anno_lines_p[i * 9 + 1 + o * 2]
        coords = anno_lines_p[i * 9 + 1 + o * 2 + 1].split(',')
        for c in range(len(coords)):
            coords[c] = int(coords[c])
        organ_dict[name] = coords
    pos_list.append(organ_dict)
for i in range(N_NEG):
    organ_dict = {}
    for o in range(4):
        name = anno_lines_n[i * 9 + 1 + o * 2]
        coords = anno_lines_n[i * 9 + 1 + o * 2 + 1].split(',')
        for c in range(len(coords)):
            coords[c] = int(coords[c])
        organ_dict[name] = coords
    neg_list.append(organ_dict)

aleph_b = open(os.path.join(root, "faces.b"), 'w')
aleph_f = open(os.path.join(root, "faces.f"), 'w')
aleph_n = open(os.path.join(root, "faces.n"), 'w')
eval_file = open(os.path.join(root, "evaluation.pl"), 'w')


def wl(f, text):
    if f == 'b':
        aleph_b.write(text + "\n")
    elif f == 'f':
        aleph_f.write(text + "\n")
    elif f == 'n':
        aleph_n.write(text + "\n")
    elif f == 'x':
        aleph_b.write(text + "\n")
        eval_file.write(text + "\n")
    elif f == 'e':
        eval_file.write(text + "\n")


wl('b', ":- use_module(library(lists)).")
wl('b', "")
wl('b', ":- modeh(1, face(+example)).")
wl('b', "")
wl('b', ":- modeb(*, contains(+example, -part)).")
wl('b', ":- modeb(*, isa(+part, #organ)).")
wl('b', ":- modeb(*, left_of(+part, +part)).")
wl('b', ":- modeb(*, top_of(+part, +part)).")
wl('b', "")
wl('b', ":- determination(face/1, contains/2).")
wl('b', ":- determination(face/1, isa/2).")
wl('b', ":- determination(face/1, left_of/2).")
wl('b', ":- determination(face/1, top_of/2).")
wl('b', "")
wl('b', ":- set(i, 5).")
wl('b', ":- set(clauselength, 30).")
wl('b', ":- set(minpos, 2).")
wl('b', "%:- set(minscore, 0).")
wl('b', "%:- set(verbosity, 0).")
wl('b', ":- set(noise, 0).")
wl('b', ":- set(nodes, 50000).")
wl('b', "")

# background knowledge
## example-organ-linkage and constellations
### pos
for example_index in range(len(pos_list)):
    wl('f', "face(p" + str(example_index) + ").")
    wl('e', "positive(p" + str(example_index) + ").")
    for k, v in pos_list[example_index].items():
        if (v[0] != -1):  # is the organ contained in the example?
            wl('x', "contains(p" + str(example_index) + ", p" + k + str(example_index) + ").")
            ke = k
            if k == 'biggest_eye' or k == 'second_eye':
                ke = 'eye'
            wl('x', "isa(p" + k + str(example_index) + ", " + ke + ").")
    for obj_k, obj_v in pos_list[example_index].items():
        for ref_k, ref_v in pos_list[example_index].items():
            constellations = get_constellations(obj_v, ref_v)
            for c in constellations:
                wl('x', c + "(p" + obj_k + str(example_index) + ", p" + ref_k + str(example_index) + ").")
### neg
for example_index in range(len(neg_list)):
    wl('n', "face(n" + str(example_index) + ").")
    wl('e', "negative(n" + str(example_index) + ").")
    for k, v in neg_list[example_index].items():
        if (v[0] != -1):  # is the organ contained in the example?
            wl('x', "contains(n" + str(example_index) + ", n" + k + str(example_index) + ").")
            ke = k
            if k == 'biggest_eye' or k == 'second_eye':
                ke = 'eye'
            wl('x', "isa(n" + k + str(example_index) + ", " + ke + ").")
    for obj_k, obj_v in neg_list[example_index].items():
        for ref_k, ref_v in neg_list[example_index].items():
            constellations = get_constellations(obj_v, ref_v)
            for c in constellations:
                wl('x', c + "(n" + obj_k + str(example_index) + ", n" + ref_k + str(example_index) + ").")

aleph_b.close()
aleph_f.close()
aleph_n.close()
eval_file.close()
