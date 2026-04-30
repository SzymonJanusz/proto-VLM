"""Build pre-processed .npz batch files for sag_refseg (ported from kdwonn/SaG build_batches.py).

Requirements (not installed by default):
  - PYTHONPATH must include external/refer/ and external/coco/PythonAPI/
  - pycocotools (pip install pycocotools or build from external/coco/PythonAPI)
  - skimage (pip install scikit-image)

Run from repo root:
  PYTHONPATH=external/refer:external/coco/PythonAPI \\
      python sag_refseg/scripts/build_batches.py -d Gref -t train

Expected output layout:
  data/refcoco/{dataset}/{split}_batch/{dataset}_{split}_{N}.npz
"""
import argparse
import os
import sys

# Resolve paths relative to the repo root (two levels above this script)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'external', 'coco', 'PythonAPI'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'external', 'refer'))

import numpy as np
import skimage.io
import skimage.transform

# Vocabulary utilities live in sag_refseg.utils (also importable from repo root)
sys.path.insert(0, _REPO_ROOT)
from sag_refseg.utils import load_vocab_dict_from_file, preprocess_sentence

try:
    from refer import REFER
    from pycocotools import mask as cocomask
except ImportError as e:
    raise ImportError(
        'Could not import refer or pycocotools. '
        'Run with: PYTHONPATH=external/refer:external/coco/PythonAPI python ...'
    ) from e


_VOCAB_FILE_DEFAULT = os.path.join(_REPO_ROOT, 'data', 'vocabulary_Gref.txt')
_IM_DIR_DEFAULT = os.path.join(_REPO_ROOT, 'data', 'coco')
_REFER_DATA_DEFAULT = os.path.join(_REPO_ROOT, 'external', 'refer', 'data')


def build_coco_batches(output_dir, dataset, setname, T, input_H, input_W,
                       coco_dir=None, refer_data=None, vocab_file=None):
    im_type = 'train2014'
    im_dir = coco_dir or _IM_DIR_DEFAULT
    refer_data_dir = refer_data or _REFER_DATA_DEFAULT
    vocab_path = vocab_file or _VOCAB_FILE_DEFAULT

    data_folder = os.path.join(output_dir, dataset, setname + '_batch')
    data_prefix = f'{dataset}_{setname}'
    os.makedirs(data_folder, exist_ok=True)

    if dataset == 'Gref':
        refer = REFER(refer_data_dir, dataset='refcocog', splitBy='google')
    elif dataset == 'unc':
        refer = REFER(refer_data_dir, dataset='refcoco', splitBy='unc')
    elif dataset == 'unc+':
        refer = REFER(refer_data_dir, dataset='refcoco+', splitBy='unc')
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    refs = [refer.Refs[r] for r in refer.Refs if refer.Refs[r]['split'] == setname]
    vocab_dict = load_vocab_dict_from_file(vocab_path)

    n_batch = 0
    for ref in refs:
        im_name = f'COCO_{im_type}_{str(ref["image_id"]).zfill(12)}'
        im = skimage.io.imread(os.path.join(im_dir, im_type, im_name + '.jpg'))
        seg = refer.Anns[ref['ann_id']]['segmentation']
        rle = cocomask.frPyObjects(seg, im.shape[0], im.shape[1])
        mask = np.max(cocomask.decode(rle), axis=2).astype(np.float32)

        if 'train' in setname:
            im_h, im_w = im.shape[:2]
            scale = min(input_H / im_h, input_W / im_w)
            resized_h = int(np.round(im_h * scale))
            resized_w = int(np.round(im_w * scale))
            im = skimage.img_as_ubyte(skimage.transform.resize(im, [resized_h, resized_w]))
            mask = skimage.transform.resize(mask, [resized_h, resized_w])

        if im.ndim == 2:
            im = np.tile(im[:, :, np.newaxis], (1, 1, 3))

        for sentence in ref['sentences']:
            print(f'saving batch {n_batch + 1}')
            sent = sentence['sent']
            text = preprocess_sentence(sent, vocab_dict, T)
            np.savez(
                file=os.path.join(data_folder, f'{data_prefix}_{n_batch}.npz'),
                text_batch=text,
                im_batch=im,
                mask_batch=(mask > 0),
                sent_batch=[sent],
                im_name_batch=im_name,
            )
            n_batch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='unc', choices=('unc', 'unc+', 'Gref'))
    parser.add_argument('-t', type=str, default='train')
    parser.add_argument('--img-size', type=int, default=480)
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(_REPO_ROOT, 'data', 'refcoco'))
    parser.add_argument('--coco-dir', type=str, default=None,
                        help='Directory containing train2014/ (default: data/coco/ under repo root)')
    parser.add_argument('--refer-data', type=str, default=None,
                        help='Path to refer/data/ directory (default: external/refer/data/)')
    parser.add_argument('--vocab-file', type=str, default=None,
                        help='Path to vocabulary_Gref.txt (default: data/vocabulary_Gref.txt)')
    args = parser.parse_args()

    build_coco_batches(
        output_dir=args.output_dir,
        dataset=args.d,
        setname=args.t,
        T=20,
        input_H=args.img_size,
        input_W=args.img_size,
        coco_dir=args.coco_dir,
        refer_data=args.refer_data,
        vocab_file=args.vocab_file,
    )
