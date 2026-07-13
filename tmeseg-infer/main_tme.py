"""
Example:
    cd ./tmeseg-infer
    /usr/bin/python3 main_tme.py \
        -d /path/to/raw_slides \
        -o /path/to/output \
        -p "*.svs" \
        -c \
        -ps 512 \
        -ins 512 \
        -nC 6 \
        -n 0 \
        -sf 0.0625 \
        -nJ 5

Output layout under -o/--save_dir:
    <save_dir>/cws_tiling/          tiles produced by cws_tiling/main_tiles.py
    <save_dir>/mask_cws512/         per-tile TME prediction
    <save_dir>/mask_ss1512/         stitched slide-level mask
    <save_dir>/mask_ss1512_post/    stitched mask after post-processing
"""
import os
import sys
import argparse
from glob import glob

from predict_slide import generate_tme
from ss1_stich import ss1_stich
from ss1_post import postprocess_tme

# cws_tiling/ is a sibling folder of this script containing the vendored
# tiling code (main_tiles.py, save_cws.py, cws_generator.py)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cws_tiling'))
import save_cws  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', dest='data_dir', required=True, help='path to raw whole-slide images')
parser.add_argument('-o', '--save_dir', dest='save_dir', required=True, help='root directory for all outputs (tiles + masks)')
parser.add_argument('-p', '--pattern', dest='file_name_pattern', default='*.svs', help='pattern in the file names')
parser.add_argument('-mpp', '--output_mpp', dest='output_mpp', type=float, default=0.22, help='tiling output mpp')
parser.add_argument('-c', '--color', dest='color_norm', help='color normalization', action='store_false')
parser.add_argument('-n', '--nfile', dest='nfile', help='the nfile-th file', default=0, type=int)
parser.add_argument('-ps', '--patch_size', dest='patch_size', help='the size of the patch', default=768, type=int)
parser.add_argument('-ins', '--input_size', dest='input_size', help='the size of the model input', default=384, type=int)
parser.add_argument('-nC', '--number_class', dest='nClass', help='how many classes to segment', default=6, type=int)
parser.add_argument('-sf', '--scale_factor', dest='scale', help='how many times to scale compared to x20', default=0.0625, type=float)
parser.add_argument('-nJ', '--number_pods', dest='nJob', help='how many pods to be used in K8s', default=32, type=int)
args = parser.parse_args()

cws_dir = os.path.join(args.save_dir, 'cws_tiling')
mask_dir = os.path.join(args.save_dir, 'mask_cws512')
ss1_dir = os.path.join(args.save_dir, 'mask_ss1512')
ss1_post_dir = os.path.join(args.save_dir, 'mask_ss1512_post')

files = sorted(glob(os.path.join(args.data_dir, args.file_name_pattern)))
njob = args.nJob
nfile = args.nfile

if len(files) <= 32:
    start_file = nfile
    end_file = nfile + 1
else:
    file_job = len(files) // njob + 1
    start_file = nfile * file_job
    end_file = nfile * file_job + file_job

######step0: make sure every slide is tiled before segmenting any of them.
# generate_tme()/ss1_stich()/postprocess_tme() below re-glob cws_dir and pick
# the file at position `i`, so cws_dir must contain the same slides, in the
# same sorted order, as `files` -- tiling only the current shard's slide(s)
# would shift that ordering and segment the wrong slide. Already-tiled slides
# (a Da*.jpg present) are skipped so re-running this script per shard doesn't
# redo tiling work.
for slide_path in files:
    slide_name = os.path.basename(slide_path)
    slide_tile_dir = os.path.join(cws_dir, slide_name)
    if glob(os.path.join(slide_tile_dir, 'Da*')):
        continue
    save_cws.run(
        opts_in={
            'output_dir': cws_dir,
            'wsi_input': slide_path,
            'tif_obj': 40,
            'cws_objective_value': 20,
            'in_mpp': None,
            'out_mpp': args.output_mpp,
            'out_mpp_target_objective': 40,
            'parallel': False,
        },
        file_name_pattern=args.file_name_pattern,
    )

for i in range(start_file, end_file):
    if i >= len(files):
        break

    ######step1: generate tme masks for tiles
    generate_tme(datapath=cws_dir, save_dir=mask_dir, file_pattern=args.file_name_pattern,
                 color_norm=args.color_norm, nfile=i, patch_size=args.patch_size,
                 patch_stride=args.patch_size*0.5, input_size=args.input_size, nClass=args.nClass)

    #######step2: stich to ss1 level
    ss1_stich(cws_folder=cws_dir, annotated_dir=mask_dir, output_dir=ss1_dir,
              nfile=i, file_pattern=args.file_name_pattern, downscale=args.scale)

    #######step3: refine ss1 mask
    postprocess_tme(cws_folder=cws_dir, ss1_dir=ss1_dir, ss1_post_dir=ss1_post_dir,
                     nfile=i, file_pattern=args.file_name_pattern)
