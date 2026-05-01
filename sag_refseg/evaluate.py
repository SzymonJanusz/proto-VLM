"""Evaluation entrypoint for sag_refseg (adapted from kdwonn/SaG eval_cma_recon.py)."""
import json
import os
import os.path as osp

import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange, reduce
from einops import einsum as ein_sum
from tqdm import tqdm

from sag_refseg.data.refer_dataset import get_test_loader
from sag_refseg.model.encoders import ImageTextEncodersRecon
from sag_refseg.model.sag_model import CrossModalAttentionRecon
from sag_refseg.option import parse_args, verify_input_args
from sag_refseg.utils import f_out_hook, load_checkpoint


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #

def mask_IU(masks, target):
    """Return (intersection, union) — same signature as inference_refer.py."""
    assert target.shape[-2:] == masks.shape[-2:]
    I = int(np.sum(np.logical_and(masks, target)))
    U = int(np.sum(np.logical_or(masks, target)))
    return I, U


def _attn_map_to_pred(attn_map, feat_map_size, label_shape, threshold):
    """Upsample slot attention map, normalise to [0,1], threshold → bool mask."""
    a = attn_map.reshape(1, feat_map_size, feat_map_size)
    a = transforms.functional.resize(a, list(label_shape)).squeeze().cpu().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return a >= threshold


def _unwrap(block):
    """Return inner fn when block is a PreNorm wrapper, else return block directly."""
    return block.fn if hasattr(block, 'fn') else block


# --------------------------------------------------------------------------- #
# Feature encoding
# --------------------------------------------------------------------------- #

def encode_data(model, data_loader, args):
    """Return (slot_a_maps, cm_a_maps) for the full dataset."""
    model.eval()

    n = len(data_loader.dataset)
    agg_depth = len(model.encoders.img_enc.set_pred_module.agg.agg_blocks)
    num_slot = model.encoders.img_enc.set_pred_module.agg.num_latents
    head = _unwrap(model.encoders.img_enc.set_pred_module.agg.agg_blocks[0][0]).heads
    head_cm = _unwrap(model.cma.attn).heads
    feat_map_cells = int(args.crop_size / 16) ** 2

    slot_a_maps = torch.zeros([n, agg_depth, num_slot, feat_map_cells],
                              requires_grad=False).cuda()
    cm_a_maps = torch.zeros([n, num_slot], requires_grad=False).cuda()

    slot_buf, cm_buf = [], []

    for data in tqdm(data_loader, desc='encode'):
        img, txt, txt_len, ids = data
        img, txt, txt_len = img.cuda(), txt.cuda(), txt_len.cuda()

        hdlr1 = _unwrap(model.encoders.img_enc.set_pred_module.agg.agg_blocks[0][0]).attn_holder.\
            register_forward_hook(f_out_hook(slot_buf))
        hdlr2 = _unwrap(model.cma.attn).attn_holder.\
            register_forward_hook(f_out_hook(cm_buf))

        with torch.no_grad():
            model.forward(img, txt, txt_len)

        slot_a = rearrange(torch.cat(slot_buf, dim=0),
                           '(depth bs h) n d -> bs depth h n d',
                           depth=agg_depth, h=head)
        slot_a = reduce(slot_a, 'bs depth h n d -> bs depth n d', 'mean')
        slot_a_maps[ids] = slot_a
        slot_buf.clear()

        cm_a = rearrange(torch.cat(cm_buf, dim=0),
                         '(bs h) n d -> bs h n d', h=head_cm)
        cm_a = reduce(cm_a, 'bs h n d -> bs n d', 'mean')
        cm_a_maps[ids] = ein_sum(cm_a, 'b b d -> b d')
        cm_buf.clear()

        hdlr1.remove()
        hdlr2.remove()

    return slot_a_maps, cm_a_maps


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def eval_seg(model, data_loader, args):
    """
    Compute cIoU and mIoU for three slot-pooling strategies:
      max  – slot with highest cross-modal attention weight
      avg  – weighted sum of all slots by cross-modal attention
      min  – slot with lowest cross-modal attention weight (diagnostic)
    Returns a dict with summary metrics and a per-sample list.
    """
    dataset = data_loader.dataset
    feat_map_size = args.crop_size // 16
    n = len(dataset)
    thr = args.pseudo_threshold

    slot_a_maps, cm_a_maps = encode_data(model, data_loader, args)

    cum_I = {'max': 0., 'avg': 0., 'min': 0.}
    cum_U = {'max': 0., 'avg': 0., 'min': 0.}
    sum_mIoU = {'max': 0., 'avg': 0., 'min': 0.}
    per_sample = []

    t = tqdm(range(n), desc='IoU')
    for i in t:
        _, raw_img_id, _, raw_label, _ = dataset.get_raw_item(i)
        cm_a = cm_a_maps[i]
        slot_a = slot_a_maps[i, -1]  # last agg layer

        sample = {'img_id': raw_img_id, 'index': i}

        for mode in ('max', 'avg', 'min'):
            if mode == 'max':
                idx = torch.argmax(cm_a)
                a_map = slot_a[idx]
            elif mode == 'avg':
                a_map = (cm_a.unsqueeze(1) * slot_a).sum(dim=0)
            else:
                idx = torch.argmin(cm_a)
                a_map = slot_a[idx]

            pred = _attn_map_to_pred(a_map, feat_map_size, raw_label.shape, thr)
            I, U = mask_IU(pred, raw_label)
            cum_I[mode] += I
            cum_U[mode] += U
            sum_mIoU[mode] += I / U if U > 0 else 0.0
            sample[f'{mode}_iou'] = float(I / U) if U > 0 else 0.0

        per_sample.append(sample)

        t.set_postfix({
            'max_cIoU': '%.2f%%' % (100.0 * cum_I['max'] / max(cum_U['max'], 1)),
            'avg_cIoU': '%.2f%%' % (100.0 * cum_I['avg'] / max(cum_U['avg'], 1)),
        })

    summary = {}
    for mode in ('max', 'avg', 'min'):
        summary[f'{mode}_cIoU'] = 100.0 * cum_I[mode] / max(cum_U[mode], 1)
        summary[f'{mode}_mIoU'] = sum_mIoU[mode] * 100.0 / n

    print('\n=== Results ===')
    for k, v in summary.items():
        print(f'  {k}: {v:.3f}%')

    return summary, per_sample


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    args = verify_input_args(parse_args())

    assert args.ckpt, '--ckpt is required for evaluation'
    assert osp.isfile(args.ckpt), f'Checkpoint not found: {args.ckpt}'

    data_loader = get_test_loader(args, split=args.data_split)

    model = CrossModalAttentionRecon(ImageTextEncodersRecon(args), args.embed_dim, args)

    if torch.cuda.is_available():
        if args.multi_gpu:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        torch.backends.cudnn.benchmark = True

    load_checkpoint(args.ckpt, model)

    with torch.no_grad():
        summary, per_sample = eval_seg(model, data_loader, args)

    # Write JSON output
    out_dir = osp.join('eval_results', 'sag_refseg')
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, f'{args.dataset}_{args.data_split}.json')
    result = {
        'dataset': args.dataset,
        'split': args.data_split,
        'ckpt': args.ckpt,
        'threshold': args.pseudo_threshold,
        'summary': summary,
        'per_sample': per_sample,
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nResults written to {out_path}')


if __name__ == '__main__':
    main()
