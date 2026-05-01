"""Training entrypoint for sag_refseg (adapted from kdwonn/SaG train_cma_recon.py)."""
import math
import os
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.cuda.amp
import torchvision.transforms as transforms
import wandb
from einops import rearrange
from tqdm import tqdm

from sag_refseg.data.refer_dataset import get_train_loader, get_test_loader
from sag_refseg.loss.cma_loss import CMA_Loss, CMA_Loss_Fast
from sag_refseg.model.encoders import ImageTextEncodersRecon
from sag_refseg.model.sag_model import CrossModalAttentionRecon
from sag_refseg.option import parse_args, verify_input_args
from sag_refseg.sync_batchnorm import convert_model, SynchronizedBatchNorm2d
from sag_refseg.utils import (
    AverageMeter, set_seed, setup_logger, save_checkpoint, load_checkpoint,
    update_training_history,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def compute_mask_IU(masks, target):
    import numpy as np
    assert target.shape[-2:] == masks.shape[-2:]
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    return I, U


def _unwrap(block):
    """Return inner fn when block is a PreNorm wrapper, else return block directly."""
    return block.fn if hasattr(block, 'fn') else block


def encode_data(model, data_loader, crop_size, img_num_embeds, embed_dim, args):
    """Encode all images and captions; collect slot and cross-modal attention maps."""
    from sag_refseg.utils import f_out_hook
    from einops import reduce, einsum as ein_sum

    model.eval()

    n = len(data_loader.dataset)
    agg_depth = len(model.encoders.img_enc.set_pred_module.agg.agg_blocks)
    num_slot = model.encoders.img_enc.set_pred_module.agg.num_latents
    head = _unwrap(model.encoders.img_enc.set_pred_module.agg.agg_blocks[0][0]).heads
    head_cm = _unwrap(model.cma.attn).heads

    slot_a_maps = torch.zeros([n, agg_depth, num_slot, int(crop_size / 16) ** 2],
                              requires_grad=False).cuda()
    cm_a_maps = torch.zeros([n, num_slot], requires_grad=False).cuda()

    slot_a_map_buf, cm_a_map_buf = [], []

    for data in tqdm(data_loader, desc='encode', leave=False):
        img, txt, txt_len, ids = data
        img, txt, txt_len = img.cuda(), txt.cuda(), txt_len.cuda()

        hdlr1 = _unwrap(model.encoders.img_enc.set_pred_module.agg.agg_blocks[0][0]).attn_holder.\
            register_forward_hook(f_out_hook(slot_a_map_buf))
        hdlr2 = _unwrap(model.cma.attn).attn_holder.\
            register_forward_hook(f_out_hook(cm_a_map_buf))

        cm_emb, img_emb, txt_emb, _, _, _ = model.forward(img, txt, txt_len)

        slot_a = rearrange(
            torch.cat(slot_a_map_buf, dim=0),
            '(depth bs h) n d -> bs depth h n d', depth=agg_depth, h=head)
        slot_a = reduce(slot_a, 'bs depth h n d -> bs depth n d', 'mean')
        slot_a_maps[ids] = slot_a
        slot_a_map_buf.clear()

        cm_a = rearrange(torch.cat(cm_a_map_buf, dim=0), '(bs h) n d -> bs h n d', h=head_cm)
        cm_a = reduce(cm_a, 'bs h n d -> bs n d', 'mean')
        cm_a_maps[ids] = ein_sum(cm_a, 'b b d -> b d')
        cm_a_map_buf.clear()

        hdlr1.remove()
        hdlr2.remove()

    return slot_a_maps, cm_a_maps


# --------------------------------------------------------------------------- #
# Training / validation
# --------------------------------------------------------------------------- #

def train(epoch, total_iter, data_loader, model, criterion, recon_criterion,
          recon_weight, optimizer, scaler, recon_warm, args, scheduler=None):
    model.train()
    if args.bn_eval:
        mods = model.module.modules() if args.multi_gpu else model.modules()
        for m in mods:
            if isinstance(m, (nn.BatchNorm2d, SynchronizedBatchNorm2d)):
                m.eval()

    losses = AverageMeter()
    losses_dict = {'cm_loss': AverageMeter(), 'recon': AverageMeter()}

    for itr, data in enumerate(data_loader):
        total_iter += 1

        if args.fast_batch:
            img, txt, txt_len, recovery, num_txts, ids = data
            img, txt, txt_len, recovery, num_txts = (
                img.cuda(), txt.cuda(), txt_len.cuda(), recovery.cuda(), num_txts.cuda())
        else:
            img, txt, txt_len, ids = data
            img, txt, txt_len = img.cuda(), txt.cuda(), txt_len.cuda()

        with torch.cuda.amp.autocast(enabled=args.amp):
            cm_feat, img_emb, txt_emb, img_feat_recon, img_feat, txt_bert = \
                model.forward(img, txt, txt_len)

            if recon_warm:
                loss, loss_dict = torch.tensor(0.0).cuda(), {}
            else:
                if args.fast_batch:
                    txt_emb, cm_feat = txt_emb[recovery], cm_feat[:, recovery, :]
                    loss, loss_dict = criterion(cm_feat, txt_emb, num_txts, txt_bert=txt_bert)
                else:
                    loss, loss_dict = criterion(cm_feat, txt_emb, img_emb, txt_bert=txt_bert)

            recon_loss = recon_criterion(img_feat_recon, img_feat.detach())
            loss_dict['recon'] = recon_loss
            loss = loss + recon_weight * recon_loss

            if total_iter < args.lr_warmup_iter:
                loss = loss * (float(total_iter) / args.lr_warmup_iter)

        losses.update(loss.item())
        for k, v in loss_dict.items():
            losses_dict[k].update(v.item() if torch.is_tensor(v) else v)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if not args.no_wandb:
            wandb.log({'iter': total_iter})
        if scheduler is not None and total_iter >= args.lr_warmup_iter:
            scheduler.step()

        if itr > 0 and (itr % args.log_step == 0 or itr + 1 == len(data_loader)):
            n_digits = int(math.ceil(math.log(len(data_loader) + 1, 10)))
            msg = 'loss: %.4f (%.4f)' % (losses.val, losses.avg)
            for k, v in losses_dict.items():
                msg += ', %s: %.4f (%.4f)' % (k.replace('_loss', ''), v.val, v.avg)
            print('[%d][%*d/%d] %s' % (epoch, n_digits, itr, len(data_loader), msg))

    return losses.avg, {k: v.avg for k, v in losses_dict.items()}, total_iter


def validation(epoch, data_loader, model, criterion, recon_criterion, recon_weight, args):
    with torch.no_grad():
        losses = AverageMeter()
        losses_dict = {'cm_loss': AverageMeter(), 'recon': AverageMeter()}

        for data in tqdm(data_loader, desc='val', leave=False):
            img, txt, txt_len, _ = data
            img, txt, txt_len = img.cuda(), txt.cuda(), txt_len.cuda()

            with torch.cuda.amp.autocast(enabled=args.amp):
                cm_feat, img_emb, txt_emb, img_feat_recon, img_feat, txt_bert = \
                    model.forward(img, txt, txt_len)
                loss, loss_dict = criterion(cm_feat, txt_emb, img_emb, txt_bert=txt_bert)
                recon_loss = recon_criterion(img_feat_recon, img_feat.detach())
                loss_dict['recon'] = recon_loss
                loss = loss + recon_weight * recon_loss

            losses.update(loss.item())
            for k, v in loss_dict.items():
                losses_dict[k].update(v.item() if torch.is_tensor(v) else v)

        msg = 'loss: %.4f (%.4f)' % (losses.val, losses.avg)
        for k, v in losses_dict.items():
            msg += ', %s: %.4f (%.4f)' % (k.replace('_loss', ''), v.val, v.avg)
        print('Epoch [%d] val: %s' % (epoch, msg))

        # Compute IoU metrics on the val split
        slot_a_maps, cm_a_maps = encode_data(
            model, data_loader,
            crop_size=args.crop_size,
            img_num_embeds=args.img_num_embeds,
            embed_dim=args.embed_dim,
            args=args,
        )

        dataset = data_loader.dataset
        feat_map_size = args.crop_size // 16
        cum_max_I = cum_max_U = cum_avg_I = cum_avg_U = 0.0
        max_mIoU = avg_mIoU = 0.0

        t = tqdm(range(len(dataset)), desc='IoU', leave=False)
        for i in t:
            _, _, _, raw_label, _ = dataset.get_raw_item(i)
            cm_a = cm_a_maps[i]

            top_idx = torch.argmax(cm_a)
            a_max = slot_a_maps[i, -1, top_idx].reshape(1, feat_map_size, feat_map_size)
            a_max = transforms.functional.resize(a_max, list(raw_label.shape)).squeeze().cpu().numpy()
            a_max = (a_max - a_max.min()) / (a_max.max() - a_max.min() + 1e-9)
            I, U = compute_mask_IU(a_max >= args.pseudo_threshold, raw_label)
            cum_max_I += I; cum_max_U += U; max_mIoU += I / U

            avg_a = (cm_a.unsqueeze(1) * slot_a_maps[i, -1]).sum(dim=0)
            avg_a = avg_a.reshape(1, feat_map_size, feat_map_size)
            avg_a = transforms.functional.resize(avg_a, list(raw_label.shape)).squeeze().cpu().numpy()
            avg_a = (avg_a - avg_a.min()) / (avg_a.max() - avg_a.min() + 1e-9)
            I, U = compute_mask_IU(avg_a >= args.pseudo_threshold, raw_label)
            cum_avg_I += I; cum_avg_U += U; avg_mIoU += I / U

            t.set_postfix({
                'max|avg': '%.3f%%|%.3f%%' % (
                    100 * (cum_max_I / cum_max_U), 100 * (cum_avg_I / cum_avg_U))
            })

        n = len(dataset)
        val_dict = {
            'max_cIoU': 100.0 * cum_max_I / cum_max_U,
            'max_mIoU': max_mIoU * 100.0 / n,
            'avg_cIoU': 100.0 * cum_avg_I / cum_avg_U,
            'avg_mIoU': avg_mIoU * 100.0 / n,
        }
        return losses.avg, {k: v.avg for k, v in losses_dict.items()}, val_dict


def update_best_score(new_score, old_score, higher_better=True):
    if old_score is None:
        return new_score, True
    if higher_better:
        return max(new_score, old_score), new_score > old_score
    return min(new_score, old_score), new_score < old_score


def warmup_backbone(model, epoch, args):
    if not (args.img_finetune and args.txt_finetune):
        return
    warm = epoch >= args.warm_epoch
    inner = model.module if args.multi_gpu else model
    if args.warm_img:
        for p in inner.encoders.img_enc.img_backbone.parameters():
            p.requires_grad = warm
    if args.warm_txt:
        for p in inner.encoders.txt_enc.bert.parameters():
            p.requires_grad = warm


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = verify_input_args(parse_args())
    set_seed(args.seed)

    out_dir = osp.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger(out_dir, name='sag_refseg')
    history_path = osp.join(out_dir, 'training_history.json')

    if not args.no_wandb:
        wandb.init(
            project='sag_refseg',
            entity='gmum',
            name=args.remark or f'sag_{args.dataset}',
            group=args.wandb_group or args.dataset,
            config=vars(args),
        )

    trn_loader = get_train_loader(args)
    val_loader = get_test_loader(args, split='val')

    model = CrossModalAttentionRecon(ImageTextEncodersRecon(args), args.embed_dim, args)

    if torch.cuda.is_available():
        if args.multi_gpu:
            model = nn.DataParallel(model)
        if args.sync_bn:
            model = convert_model(model)
        model = model.cuda()
        cudnn.benchmark = True

    if not args.no_wandb:
        wandb.watch(model, log_freq=1000, log='gradients')

    recon_criterion = nn.MSELoss()

    criterion = (CMA_Loss_Fast if args.fast_batch else CMA_Loss)(
        margin=args.margin,
        criterion=args.cma_criterion,
        mining=args.cma_mining,
        detach_target=args.cma_detach_target,
        detach_img_target=args.cma_detach_img_target,
        i_t_loss=None,
        i_t_weight=args.i_t_weight,
        temperature=args.info_temperature,
        cm_i_weight=args.cm_i_weight,
        size_p_loss=None,
        size_p_weight=args.size_p_weight,
    )
    val_criterion = CMA_Loss(
        margin=args.margin,
        criterion=args.cma_criterion,
        mining=args.cma_mining,
        detach_target=args.cma_detach_target,
        detach_img_target=args.cma_detach_img_target,
        i_t_weight=args.i_t_weight,
        temperature=args.info_temperature,
        cm_i_weight=args.cm_i_weight,
        size_p_weight=args.size_p_weight,
    )

    inner = model.module if args.multi_gpu else model
    param_groups = [
        {'params': inner.cma.parameters(), 'lr': args.lr},
        {'params': inner.decoder.parameters(), 'lr': args.lr},
        {'params': list(set(inner.encoders.img_enc.parameters()).difference(
            set(inner.encoders.img_enc.img_backbone.parameters()))),
         'lr': args.lr * args.img_spm_lr_scale},
        {'params': inner.encoders.img_enc.img_backbone.parameters(),
         'lr': args.lr * args.img_lr_scale},
        {'params': list(inner.encoders.txt_enc.parameters()),
         'lr': args.lr * args.txt_lr_scale},
    ]

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                     weight_decay=args.weight_decay, amsgrad=True)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr,
                                      weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')

    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(trn_loader) * args.num_epochs)
    elif args.lr_scheduler == 'multi_step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_milestones, gamma=args.lr_step_gamma)
    elif args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_step_gamma)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    start_epoch = 0
    total_iter = 0
    best_iou = None

    if args.ckpt:
        start_epoch = load_checkpoint(args.ckpt, model, optimizer, scheduler)
        logger.info(f'Resumed from {args.ckpt}, epoch {start_epoch}')

    for epoch in range(start_epoch, args.num_epochs):
        warmup_backbone(model, epoch, args)
        recon_weight = args.recon_weight if epoch >= args.wo_recon_epoch else 0.0

        trn_loss, trn_losses, total_iter = train(
            epoch, total_iter, trn_loader, model, criterion, recon_criterion,
            recon_weight, optimizer, scaler,
            recon_warm=(epoch < args.recon_warm_epoch),
            args=args,
            scheduler=scheduler if args.lr_scheduler == 'cosine' else None,
        )

        if epoch % args.eval_epoch == 0:
            val_loss, val_losses, val_dict = validation(
                epoch, val_loader, model, val_criterion,
                recon_criterion, recon_weight, args,
            )
        else:
            val_loss, val_losses, val_dict = None, {}, {}

        if args.lr_scheduler != 'cosine' and scheduler is not None:
            scheduler.step()

        metrics = {
            'trn_loss': trn_loss,
            **{f'trn_{k}': v for k, v in trn_losses.items()},
            **({'val_loss': val_loss} if val_loss is not None else {}),
            **{f'val_{k}': v for k, v in val_losses.items()},
            **val_dict,
        }
        update_training_history(history_path, epoch, metrics)

        if not args.no_wandb:
            wandb.log({'epoch': epoch, 'LR': optimizer.param_groups[0]['lr'],
                       **metrics}, step=total_iter)

        logger.info(f'Epoch {epoch}: trn_loss={trn_loss:.4f} | {val_dict}')

        is_best = False
        if val_dict:
            best_iou, is_best = update_best_score(
                val_dict.get('avg_mIoU', 0.0), best_iou, higher_better=True)

        save_checkpoint(model, vars(args), out_dir, epoch,
                        is_best=is_best, filename='latest.pth')

    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
