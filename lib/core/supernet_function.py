import os
import time
import torch
import logging
import torchvision

import numpy as np
import torch.nn.functional as F

from collections import OrderedDict
from copy import deepcopy
from lib.utils.loss import cross_entropy_loss_with_soft_target
from lib.utils.helpers import AverageMeter, accuracy, reduce_tensor

def get_prob(args, best_children_pool, CHOICE_NUM=6):
    if args.how_to_prob == 'even' or (args.how_to_prob == 'teacher' and len(best_children_pool) == 0):
        return None
    elif args.how_to_prob == 'pre_prob':
        return args.pre_prob
    elif args.how_to_prob == 'teacher':
        op_dict = {}
        for i in range(CHOICE_NUM):
            op_dict[i]=0
        for item in best_children_pool:
            cand = item[3]
            for block in cand:
                for op in block:
                    op_dict[op]+=1
        sum_op = 0
        for i in range(CHOICE_NUM):
            sum_op = sum_op + op_dict[i]
        prob = []
        for i in range(CHOICE_NUM):
            prob.append(float(op_dict[i])/sum_op)
        del op_dict, sum_op
        return prob

def get_cand_with_prob(CHOICE_NUM, prob=None, sta_num=(4,4,4,4,4)):
    if prob is None:
        get_random_cand = [np.random.choice(CHOICE_NUM, item).tolist() for item in sta_num]
    else:
        get_random_cand = [np.random.choice(CHOICE_NUM, item, prob).tolist() for item in sta_num]
    # print(get_random_cand)
    return get_random_cand

def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args, sta_num=None,
        est=None, val_loader=None, best_children_pool=None, logger=None, saved_val_images=None, saved_val_labels=None,
        lr_scheduler=None, saver=None, output_dir='', use_amp=False, model_ema=None, CHOICE_NUM=4):
    if args.prefetcher and args.mixup > 0 and loader.mixup_enabled:
        if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
            loader.mixup_enabled = False

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    kd_losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    def get_model(model):
        try:
            return model.module
        except:
            return model

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if args.tiny:
            input = input.cuda()
            target = target.cuda()
        elif not args.prefetcher:
            input = input.cuda()
            target = target.cuda()

        prob = get_prob(args, best_children_pool)
        get_random_cand = get_cand_with_prob(CHOICE_NUM, prob, sta_num=sta_num)
        # add head and tail
        get_random_cand.insert(0, [0])
        get_random_cand.append([0])

        cand_flops = est.get_flops(get_random_cand)

        if epoch > args.meta_sta_epoch and batch_idx > 0 and batch_idx % args.update_iter == 0:
            if args.update_1nd:
                slice = args.slice
                x = input[:slice]
                teacher_outputt = model(x, cand)
                outputt = model(x, get_random_cand)
                soft_labell = F.softmax(teacher_outputt, dim=1)
                kd_tea = cross_entropy_loss_with_soft_target(soft_labell, outputt)
                optimizer.zero_grad()
                kd_tea.backward()
                optimizer.step()
                del teacher_outputt, outputt, soft_label, kd_tea
            elif args.update_2nd:
                slice = args.slice
                x = deepcopy(input[:slice].clone().detach())

                if len(best_children_pool) > 0:
                    if args.pick_method == 'top1':
                        meta_value, cand = 1, sorted(best_children_pool, reverse=True)[0][3]
                    elif args.pick_method == 'meta':
                        meta_value, cand_idx, cand = -1000000000, -1, None
                        for now_idx, item in enumerate(best_children_pool):
                            inputx = item[4]
                            output = F.softmax(model(inputx, get_random_cand), dim=1)
                            weight = get_model(model).forward_meta(output - item[5])
                            if weight > meta_value:
                                meta_value = weight  # deepcopy(torch.nn.functional.sigmoid(weight))
                                cand_idx = now_idx
                                cand = best_children_pool[cand_idx][3]
                        assert cand is not None
                        meta_value = torch.nn.functional.sigmoid(-weight)
                    else:
                        raise ValueError('Method Not supported')

                u_output = model(x, get_random_cand)
                u_teacher_output = model(x, cand)
                u_soft_label = F.softmax(u_teacher_output, dim=1)
                kd_loss = meta_value * cross_entropy_loss_with_soft_target(u_output, u_soft_label)
                optimizer.zero_grad()

                grad_1 = torch.autograd.grad(kd_loss,
                                             get_model(model).rand_parameters(get_random_cand),
                                             create_graph=True)

                def raw_sgd(w, g):
                    return g * optimizer.param_groups[-1]['lr'] + w

                students_weight = [raw_sgd(p, grad_item)
                                   for p, grad_item in zip(get_model(model).rand_parameters(get_random_cand), grad_1)]

                # update student weights
                for weight, grad_item in zip(get_model(model).rand_parameters(get_random_cand), grad_1):
                    weight.grad = grad_item
                torch.nn.utils.clip_grad_norm_(get_model(model).rand_parameters(get_random_cand), 1)
                optimizer.step()
                for weight, grad_item in zip(get_model(model).rand_parameters(get_random_cand), grad_1):
                    del weight.grad

                held_out_x = input[slice:slice * 2].clone()
                output_2 = model(held_out_x, get_random_cand)
                valid_loss = loss_fn(output_2, target[slice:slice * 2])
                optimizer.zero_grad()

                grad_student_val = torch.autograd.grad(valid_loss,
                                                       get_model(model).rand_parameters(get_random_cand),
                                                       retain_graph=True)

                grad_teacher = torch.autograd.grad(students_weight[0],
                                                   get_model(model).rand_parameters(cand, args.pick_method == 'meta'),
                                                   grad_outputs=grad_student_val)

                # update teacher model
                for weight, grad_item in zip(get_model(model).rand_parameters(cand, args.pick_method == 'meta'),
                                             grad_teacher):
                    weight.grad = grad_item
                torch.nn.utils.clip_grad_norm_(
                    get_model(model).rand_parameters(get_random_cand, args.pick_method == 'meta'), 1)
                optimizer.step()
                for weight, grad_item in zip(get_model(model).rand_parameters(cand, args.pick_method == 'meta'),
                                             grad_teacher):
                    del weight.grad

                for item in students_weight:
                    del item
                del grad_teacher, grad_1, grad_student_val, x, held_out_x
                del valid_loss, kd_loss, u_soft_label, u_output, u_teacher_output, output_2

            else:
                raise ValueError("Must 1nd or 2nd update teacher weights")

        # get_best_teacher
        if len(best_children_pool) > 0:
            if args.pick_method == 'top1':
                meta_value, cand = 0.5, sorted(best_children_pool, reverse=True)[0][3]
            elif args.pick_method == 'meta':
                meta_value, cand_idx, cand = -1000000000, -1, None
                for now_idx, item in enumerate(best_children_pool):
                    inputx = item[4]
                    output = F.softmax(model(inputx, get_random_cand), dim=1)
                    weight = get_model(model).forward_meta(output - item[5])
                    if weight > meta_value:
                        meta_value = weight  # deepcopy(torch.nn.functional.sigmoid(weight))
                        cand_idx = now_idx
                        cand = best_children_pool[cand_idx][3]
                assert cand is not None
                meta_value = torch.nn.functional.sigmoid(-weight)
            else:
                raise ValueError('Method Not supported')

        if len(best_children_pool) == 0:
            output = model(input, get_random_cand)
            loss = loss_fn(output, target)
            kd_loss = loss
        elif epoch <= args.meta_sta_epoch:
            output = model(input, get_random_cand)
            loss = loss_fn(output, target)
        else:
            output = model(input, get_random_cand)
            with torch.no_grad():
                teacher_output = model(input, cand).detach()
                soft_label = F.softmax(teacher_output, dim=1)
            kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
            valid_loss = loss_fn(output, target)
            loss = (meta_value * kd_loss + (2 - meta_value) * valid_loss) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        if not args.distributed:
            reduced_loss = loss.data
        else:
            # losses_m.update(loss.item(), input.size(0))
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            prec1 = reduce_tensor(prec1, args.world_size)
            prec5 = reduce_tensor(prec5, args.world_size)

        # best_children_pool = sorted(best_children_pool, reverse=True)
        if epoch > args.meta_sta_epoch and ((len(best_children_pool) < args.pool_size) or (prec1 > best_children_pool[-1][1] + 5) or (prec1 > best_children_pool[-1][1] and cand_flops < best_children_pool[-1][2])):
            val_prec1 = prec1
            training_data = deepcopy(input[:args.slice].detach())
            if len(best_children_pool) == 0:
                features = deepcopy(output[:args.slice].detach())
            else:
                features = deepcopy(teacher_output[:args.slice].detach())
            best_children_pool.append(
                (val_prec1, prec1, cand_flops, get_random_cand, training_data, F.softmax(features, dim=1)))
            best_children_pool = sorted(best_children_pool, reverse=True)

        if len(best_children_pool) > args.pool_size:
            best_children_pool = sorted(best_children_pool, reverse=True)
            del best_children_pool[-1]

        # del valid_loss
        # backward

        torch.cuda.synchronize()

        losses_m.update(reduced_loss.item(), input.size(0))
        kd_losses_m.update(kd_loss.item(), input.size(0))
        prec1_m.update(prec1.item(), output.size(0))
        prec5_m.update(prec5.item(), output.size(0))

        # del kd_loss
        # del output
        # del reduced_loss

        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # print(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.local_rank == 0:
                logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'KD-Loss: {kd_loss.val:>9.6f} ({kd_loss.avg:>6.4f})  '
                    'Prec@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Prec@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        kd_loss=kd_losses_m,
                        top1=prec1_m,
                        top5=prec5_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(
                model, optimizer, args, epoch, model_ema=model_ema, batch_idx=batch_idx)

        end = time.time()

    if args.local_rank == 0:
        for idx, i in enumerate(best_children_pool):
            logger.info("No.{} {}".format(idx, i[:4]))

    return OrderedDict([('loss', losses_m.avg)]), best_children_pool


def validate(model, loader, loss_fn, args, log_suffix='', CHOICE_NUM=4, sta_num=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1

    get_random_cand = get_cand_with_prob(CHOICE_NUM, None, sta_num=sta_num)
    # add head and tail
    get_random_cand.insert(0, [0])
    get_random_cand.append([0])

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            output = model(input, get_random_cand)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                prec1 = reduce_tensor(prec1, args.world_size)
                prec5 = reduce_tensor(prec5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            prec1_m.update(prec1.item(), output.size(0))
            prec5_m.update(prec5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                logging.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Prec@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Prec@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx,
                        batch_time=batch_time_m, loss=losses_m,
                        top1=prec1_m, top5=prec5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('prec1', prec1_m.avg), ('prec5', prec5_m.avg)])

    return metrics
