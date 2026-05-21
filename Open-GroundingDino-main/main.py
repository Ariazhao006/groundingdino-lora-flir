# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import BestMetricHolder, ModelEma
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch

from groundingdino.util.utils import clean_state_dict
from util.lora import (
    count_trainable_parameters,
    inject_lora_by_module_name,
    mark_only_lora_trainable,
)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument("--datasets", type=str, required=True, help='path to datasets json')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def _collect_odvg_metas(dataset):
    """Collect ODVG metas for per-image weighting."""
    if hasattr(dataset, "metas"):
        return list(dataset.metas)
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        metas = []
        for sub_dataset in dataset.datasets:
            metas.extend(_collect_odvg_metas(sub_dataset))
        return metas
    return []


def _build_weighted_train_sampler(dataset_train, args, logger):
    """
    Build per-image weighted sampler using ODVG detection metadata.
    This is intended to upweight images with more/smaller objects.
    """
    metas = _collect_odvg_metas(dataset_train)
    if len(metas) != len(dataset_train):
        logger.warning(
            "Weighted sampler fallback: metas length {} != dataset length {}.".format(
                len(metas), len(dataset_train)
            )
        )
        return torch.utils.data.RandomSampler(dataset_train)

    small_area_thr = float(getattr(args, "sampler_small_area_thr", 32 * 32))
    small_object_boost = float(getattr(args, "sampler_small_object_boost", 1.0))
    box_count_boost = float(getattr(args, "sampler_box_count_boost", 0.3))
    box_count_ref = float(getattr(args, "sampler_box_count_ref", 6.0))
    min_weight = float(getattr(args, "sampler_min_weight", 1.0))
    max_weight = float(getattr(args, "sampler_max_weight", 3.0))
    replacement = bool(getattr(args, "sampler_replacement", True))
    num_samples = int(getattr(args, "sampler_num_samples", len(metas)))

    weights = []
    for meta in metas:
        instances = meta.get("detection", {}).get("instances", [])
        obj_count = len(instances)
        small_count = 0
        for inst in instances:
            bbox = inst.get("bbox", None)
            if bbox is None or len(bbox) < 4:
                continue
            area = float(bbox[2]) * float(bbox[3])
            if area <= small_area_thr:
                small_count += 1

        obj_ratio = min(float(obj_count) / max(box_count_ref, 1.0), 1.0)
        small_ratio = float(small_count) / max(obj_count, 1)

        weight = 1.0 + box_count_boost * obj_ratio + small_object_boost * small_ratio
        weight = max(min_weight, min(max_weight, weight))
        weights.append(weight)

    logger.info(
        "Weighted sampler enabled: min_w={:.3f}, max_w={:.3f}, mean_w={:.3f}, num_samples={}".format(
            float(np.min(weights)),
            float(np.max(weights)),
            float(np.mean(weights)),
            num_samples,
        )
    )
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=replacement,
    )


def main(args):
    

    utils.setup_distributed(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")

    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))

    with open(args.datasets) as f:
        dataset_meta = json.load(f)
    if args.use_coco_eval:
        args.coco_val_path = dataset_meta["val"][0]["anno"]

    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    logger.debug("build model ... ...")
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    logger.debug("build model, done.")

    # freeze some layers
    if args.freeze_keywords is not None:
        for name, parameter in model.named_parameters():
            for keyword in args.freeze_keywords:
                if keyword in name:
                    parameter.requires_grad_(False)
                    break

    # optional LoRA injection for parameter-efficient finetuning
    if getattr(args, "use_lora", False):
        include_patterns = getattr(args, "lora_include_patterns", [])
        target_linear_names = getattr(args, "lora_target_linear_names", None)
        replaced_modules = inject_lora_by_module_name(
            model,
            include_patterns=include_patterns,
            rank=getattr(args, "lora_rank", 8),
            alpha=getattr(args, "lora_alpha", 16),
            dropout=getattr(args, "lora_dropout", 0.0),
            target_linear_names=target_linear_names,
        )
        logger.info(
            "LoRA enabled. Replaced {} linear layers.".format(len(replaced_modules))
        )
        logger.info(
            "LoRA module list:\n{}".format(
                json.dumps(replaced_modules, indent=2, ensure_ascii=False)
            )
        )
        if getattr(args, "lora_only_trainable", True):
            trainable_names = mark_only_lora_trainable(
                model, also_train_keywords=getattr(args, "lora_also_train_keywords", [])
            )
            logger.info(
                "LoRA-only trainable params: {}".format(len(trainable_names))
            )

    total_params, trainable_params = count_trainable_parameters(model)
    logger.info(
        "params total: {}, trainable: {}, trainable_ratio: {:.4f}".format(
            total_params, trainable_params, trainable_params / max(total_params, 1)
        )
    )
    logger.info(
        "params after freezing/lora:\n"
        + json.dumps(
            {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2
        )
    )

    model.to(device)
    model_without_ddp = model
    ema_m = None
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model._set_static_graph()
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    logger.info('number of trainable params:'+str(n_parameters))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    logger.debug("build dataset ... ...")
    if not args.eval:
        num_of_dataset_train = len(dataset_meta["train"])
        if num_of_dataset_train == 1:
            dataset_train = build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][0])
        else:
            from torch.utils.data import ConcatDataset
            dataset_train_list = []
            for idx in range(len(dataset_meta["train"])):
                dataset_train_list.append(build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][idx]))
            dataset_train = ConcatDataset(dataset_train_list)
        logger.debug("build dataset, done.")
        logger.debug(f'number of training dataset: {num_of_dataset_train}, samples: {len(dataset_train)}')

    dataset_val = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])

    sampler_mode = str(getattr(args, "train_sampler_mode", "random")).lower()
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        if not args.eval:
            if sampler_mode == "weighted":
                logger.warning("train_sampler_mode=weighted is not supported in distributed mode, fallback to distributed random sampler.")
            sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if not args.eval:
            if sampler_mode == "weighted":
                sampler_train = _build_weighted_train_sampler(dataset_train, args, logger)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers)

    data_loader_val = DataLoader(dataset_val, 4, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)


        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

    # IMPORTANT:
    # Initialize/load EMA only after resume/pretrain weights are loaded.
    # Otherwise EMA may start from random init and stay far behind.
    if getattr(args, "use_ema", False):
        ema_m = ModelEma(model_without_ddp, decay=args.ema_decay)
        if args.resume and 'checkpoint' in locals() and checkpoint.get('model_ema', None) is not None:
            ema_m.module.load_state_dict(clean_state_dict(checkpoint['model_ema']), strict=False)
            logger.info("EMA enabled. decay={}, resumed from checkpoint model_ema".format(args.ema_decay))
        else:
            ema_m.set(model_without_ddp)
            logger.info("EMA enabled. decay={}, initialized from current model weights".format(args.ema_decay))

 
    
    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return
    
 
    
    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=getattr(args, "use_ema", False))
    best_aps_holder = BestMetricHolder(use_ema=getattr(args, "use_ema", False))
    early_stop_enabled = getattr(args, "enable_early_stop", False)
    early_stop_patience = int(getattr(args, "early_stop_patience", 4))
    early_stop_min_delta = float(getattr(args, "early_stop_min_delta", 0.001))
    early_stop_warmup_epochs = int(getattr(args, "early_stop_warmup_epochs", 0))
    early_stop_best = float("-inf")
    early_stop_bad_epochs = 0
    should_stop = False
    if early_stop_enabled:
        logger.info(
            "Early stop enabled: patience={}, min_delta={}, warmup_epochs={}".format(
                early_stop_patience, early_stop_min_delta, early_stop_warmup_epochs
            )
        )

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if ema_m is not None:
                    weights['model_ema'] = ema_m.module.state_dict()

                utils.save_on_master(weights, checkpoint_path)
                
        # eval
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        map_regular = test_stats['coco_eval_bbox'][0]
        aps_regular = test_stats['coco_eval_bbox'][3]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest:
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        _isbest_aps = best_aps_holder.update(aps_regular, epoch, is_ema=False)
        if _isbest_aps:
            checkpoint_path = output_dir / 'checkpoint_best_aps_regular.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        ema_log_stats = {}
        if ema_m is not None and epoch >= getattr(args, "ema_epoch", 0):
            ema_test_stats, _ = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            ema_map = ema_test_stats['coco_eval_bbox'][0]
            ema_aps = ema_test_stats['coco_eval_bbox'][3]
            _isbest_ema = best_map_holder.update(ema_map, epoch, is_ema=True)
            if _isbest_ema:
                checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            _isbest_ema_aps = best_aps_holder.update(ema_aps, epoch, is_ema=True)
            if _isbest_ema_aps:
                checkpoint_path = output_dir / 'checkpoint_best_aps_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            ema_log_stats = {f'ema_test_{k}': v for k, v in ema_test_stats.items()}

        early_stop_stats = {}
        if early_stop_enabled:
            if epoch >= early_stop_warmup_epochs:
                if map_regular > early_stop_best + early_stop_min_delta:
                    early_stop_best = map_regular
                    early_stop_bad_epochs = 0
                else:
                    early_stop_bad_epochs += 1
                if early_stop_bad_epochs >= early_stop_patience:
                    should_stop = True
            early_stop_stats = {
                "early_stop_best_ap": early_stop_best if early_stop_best != float("-inf") else None,
                "early_stop_bad_epochs": early_stop_bad_epochs,
                "early_stop_should_stop": should_stop,
            }

        if args.distributed:
            stop_flag = torch.tensor(1 if should_stop else 0, device=device)
            torch.distributed.all_reduce(stop_flag, op=torch.distributed.ReduceOp.MAX)
            should_stop = stop_flag.item() > 0

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            **ema_log_stats,
            **early_stop_stats,
            'best_map_summary': best_map_holder.summary(),
            'best_aps_summary': best_aps_holder.summary(),
        }


        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

        if should_stop:
            logger.info("Early stopping triggered at epoch {}.".format(epoch + 1))
            break
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if args.output_dir and utils.is_main_process():
        best_summary = {
            'best_map': best_map_holder.summary(),
            'best_aps': best_aps_holder.summary(),
        }
        with (output_dir / "best_summary.json").open("w") as f:
            json.dump(best_summary, f, indent=2)
        preferred = "checkpoint_best_regular.pth"
        preferred_metric = best_summary["best_map"].get("best_res", 0.0)
        if getattr(args, "use_ema", False):
            regular_metric = best_summary["best_map"].get("regular_best_res", 0.0)
            ema_metric = best_summary["best_map"].get("ema_best_res", 0.0)
            preferred = "checkpoint_best_ema.pth" if ema_metric > regular_metric else "checkpoint_best_regular.pth"
            preferred_metric = max(regular_metric, ema_metric)
        with (output_dir / "preferred_checkpoint.json").open("w") as f:
            json.dump(
                {"checkpoint": preferred, "metric": preferred_metric, "selector": "best_map"},
                f,
                indent=2,
            )

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
