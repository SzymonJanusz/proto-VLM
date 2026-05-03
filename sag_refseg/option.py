"""Argument parser for sag_refseg (ported from kdwonn/SaG option.py + local additions)."""
import os
import json
import argparse

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the parser as a factory so we can call set_defaults() safely
def make_parser():
    p = argparse.ArgumentParser(description='sag_refseg: Shatter and Gather for RefSeg')

    # ── Local additions ──────────────────────────────────────────────────────
    p.add_argument('--config', type=str, default='',
                   help='JSON config file; values set argparse defaults (CLI overrides)')
    p.add_argument('--data_root', type=str, default='./data/refcoco',
                   help='Root dir containing Gref/, unc/, unc+/ sub-dirs')
    p.add_argument('--dataset', type=str, default='Gref', choices=('Gref', 'unc', 'unc+'))
    p.add_argument('--out_dir', type=str, default='./checkpoints/sag_refseg',
                   help='Directory for checkpoints and logs')

    # ── Names, paths, logging ────────────────────────────────────────────────
    p.add_argument('--data_name', default='coco', choices=('coco', 'phrasecut'))
    p.add_argument('--data_path', default='')
    p.add_argument('--data_split', default='train')
    p.add_argument('--vocab_path', default=CUR_DIR + '/vocab/')
    p.add_argument('--log_step', default=10, type=int)
    p.add_argument('--log_dir', default='')

    # ── Data ────────────────────────────────────────────────────────────────
    p.add_argument('--word_dim', default=300, type=int)
    p.add_argument('--workers', default=4, type=int)
    p.add_argument('--crop_size', default=384, type=int)
    p.add_argument('--use_aug', action='store_true')

    # ── Model ────────────────────────────────────────────────────────────────
    p.add_argument('--img_backbone', default='vit_small_patch16_384')
    p.add_argument('--embed_dim', default=512, type=int)
    p.add_argument('--margin', default=0.1, type=float)
    p.add_argument('--dropout', default=0.1, type=float)

    # ── Attention ────────────────────────────────────────────────────────────
    p.add_argument('--img_num_embeds', default=36, type=int)
    p.add_argument('--txt_num_embeds', default=1, type=int)

    # ── Training / optimizer ─────────────────────────────────────────────────
    p.add_argument('--img_finetune', action='store_true')
    p.add_argument('--txt_finetune', action='store_true')
    p.add_argument('--num_epochs', default=50, type=int)
    p.add_argument('--batch_size', default=32, type=int)
    p.add_argument('--grad_clip', default=0.1, type=float)
    p.add_argument('--weight_decay', default=1e-4, type=float)
    p.add_argument('--lr', default=1e-5, type=float)
    p.add_argument('--ckpt', default='', type=str, metavar='PATH')
    p.add_argument('--eval_on_gpu', action='store_true')
    p.add_argument('--warm_epoch', default=0, type=int)
    p.add_argument('--remark', type=str, default='')
    p.add_argument('--wandb_group', type=str, default='sag_refseg')
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--lr_scheduler', type=str, default='cosine')
    p.add_argument('--lr_milestones', nargs='+', type=int)
    p.add_argument('--lr_step_gamma', type=float, default=0.1)
    p.add_argument('--lr_step_size', type=int, default=20)
    p.add_argument('--warm_txt', action='store_true')
    p.add_argument('--warm_img', action='store_true')
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--sync_bn', action='store_true')
    p.add_argument('--fast_batch', action='store_true')
    p.add_argument('--num_texts', default=0, type=int)
    p.add_argument('--semi_hard_triplet', action='store_true')
    p.add_argument('--img_spm_lr_scale', type=float, default=1.0)
    p.add_argument('--txt_lr_scale', type=float, default=1.0)
    p.add_argument('--img_lr_scale', type=float, default=0.1)
    p.add_argument('--optimizer', type=str, default='adamw')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--lr_warmup_iter', type=int, default=5000)
    p.add_argument('--bn_eval', action='store_true')

    # ── Aggregator ───────────────────────────────────────────────────────────
    p.add_argument('--agg_query_self_attns', type=int, default=0)
    p.add_argument('--agg_self_per_cross_attn', type=int, default=1)
    p.add_argument('--agg_self_before_cross_attn', type=int, default=0)
    p.add_argument('--agg_depth', type=int, default=6)
    p.add_argument('--agg_cross_head', type=int, default=4)
    p.add_argument('--agg_cross_dim', type=int, default=256)
    p.add_argument('--agg_residual', action='store_true')
    p.add_argument('--agg_latent_head', type=int, default=4)
    p.add_argument('--agg_latent_dim', type=int, default=128)
    p.add_argument('--agg_last_fc', action='store_true')
    p.add_argument('--agg_input_dim', type=int, default=512)
    p.add_argument('--agg_query_dim', type=int, default=512)
    p.add_argument('--agg_pre_norm', action='store_true')
    p.add_argument('--agg_post_norm', action='store_true')
    p.add_argument('--agg_activation', type=str, default='gelu')
    p.add_argument('--agg_last_ln', action='store_true')
    p.add_argument('--agg_weight_sharing', action='store_true')
    p.add_argument('--agg_ff_mult', type=float, default=4)
    p.add_argument('--agg_xavier_init', action='store_true')
    p.add_argument('--agg_more_dropout', action='store_true')
    p.add_argument('--agg_thin_ff', action='store_true')
    p.add_argument('--agg_first_order', action='store_true')
    p.add_argument('--agg_pos_enc', type=str, default='sine')
    p.add_argument('--agg_gru', action='store_true')
    p.add_argument('--agg_cross_attn_type', default='slot', choices=('slot', 'transformer'))
    p.add_argument('--agg_gumbel_attn', action='store_true')
    p.add_argument('--agg_gumbel_last', action='store_true')
    p.add_argument('--agg_query_slot', action='store_true')
    p.add_argument('--agg_query_type', default='entity', choices=('query', 'random', 'entity'))
    p.add_argument('--cascade_factor', type=int, default=3)
    p.add_argument('--agg_var_scaling', type=float, default=1)
    p.add_argument('--decoder_normalizer', type=str, default='softmax')
    p.add_argument('--recon_decoder', type=str, default='mlp')
    p.add_argument('--decoder_self_attn', action='store_true')
    p.add_argument('--decoder_pos_enc', default='learned', choices=('learned', 'sine'))
    p.add_argument('--slot_cond', action='store_true')

    # ── Encoder options ──────────────────────────────────────────────────────
    p.add_argument('--txt_pooling', default='cls', choices=('cls', 'max', 'avg'))
    p.add_argument('--txt_l2', action='store_true')
    p.add_argument('--img_res_pool', default='avg', choices=('avg', 'max'))
    p.add_argument('--text_no_dropout', action='store_true')

    # ── Pseudo-labelling ─────────────────────────────────────────────────────
    p.add_argument('--pseudo_threshold', default=0.5, type=float)

    # ── Cross-modal attention ────────────────────────────────────────────────
    p.add_argument('--cma_heads', default=4, type=int)
    p.add_argument('--cma_head_dim', default=256, type=int)
    p.add_argument('--cma_criterion', type=str, default='info_nce')
    p.add_argument('--cma_mining', type=str, default='hard')
    p.add_argument('--cma_detach_target', action='store_true')
    p.add_argument('--cma_detach_img_target', action='store_true')
    p.add_argument('--cma_self_attn', action='store_true')
    p.add_argument('--cma_last_fc', action='store_true')
    p.add_argument('--cma_qk_norm', action='store_true')
    p.add_argument('--cma_last_mlp', action='store_true')

    # ── Loss configuration ───────────────────────────────────────────────────
    p.add_argument('--i_t_weight', default=0.0, type=float)
    p.add_argument('--info_temperature', default=1.0, type=float)
    p.add_argument('--gumbel_tau', default=1.0, type=float)
    p.add_argument('--cm_i_weight', default=1.0, type=float)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--recon_weight', type=float, default=1.0)
    p.add_argument('--num_layers', type=int, default=12)
    p.add_argument('--recon_warm_epoch', type=int, default=0)
    p.add_argument('--wo_recon_epoch', type=int, default=0)
    p.add_argument('--amap_save', action='store_true')
    p.add_argument('--agg_1x1_mlp', action='store_true')
    p.add_argument('--info_txt_l2', action='store_true')
    p.add_argument('--pre_bertemb', action='store_true')
    p.add_argument('--eval_epoch', default=1, type=int)
    p.add_argument('--size_p_weight', default=0, type=float)
    p.add_argument('--size_gamma', default=0.01, type=float)
    p.add_argument('--size_penalty', default=5, type=float)
    p.add_argument('--save_head_map', action='store_true')
    p.add_argument('--vis_samples', type=int, default=8,
                   help='Number of val images to log to W&B per epoch (0 = off)')

    return p


def parse_args(argv=None):
    """Two-pass parse: JSON config sets defaults, CLI overrides them."""
    p = make_parser()
    # First pass: extract --config only
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        with open(pre.config) as f:
            cfg = json.load(f)
        # Convert booleans stored as True in JSON to proper store_true defaults
        p.set_defaults(**{k: v for k, v in cfg.items() if hasattr(pre, k)})
    # Second pass: full parse with updated defaults
    return p.parse_args(argv)


def verify_input_args(args):
    if not args.data_path:
        args.data_path = os.path.join(args.data_root, args.dataset)
    if not args.log_dir:
        args.log_dir = os.path.join(args.out_dir, args.dataset)
    if args.agg_query_slot:
        args.agg_query_type = 'entity'
    if args.agg_query_type == 'entity':
        assert args.cascade_factor is not None, '--cascade_factor required for entity query type'
    assert not (args.agg_gumbel_attn and args.agg_gumbel_last)
    return args


if __name__ == '__main__':
    args = verify_input_args(parse_args())
    import pprint
    pprint.pprint(vars(args))
