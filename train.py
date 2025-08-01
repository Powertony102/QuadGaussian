# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from lpipsPyTorch import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb_args=None):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, wandb_args)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    optim_start = torch.cuda.Event(enable_timing = True)  # 新增
    optim_end = torch.cuda.Event(enable_timing = True)    # 新增
    total_time = 0.0  # 新增
    total_kernel_time = 0.0  # 新增，累计 kernel time

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        kernel_time = render_pkg.get("kernel_times", None)  # 新增，获取 kernel time
        if kernel_time is not None:
            total_kernel_time += float(kernel_time)

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()
        optim_start.record()  # 新增，优化器步骤前

        # Optimizer step
        if iteration < opt.iterations:
            gaussians.exposure_optimizer.step()
            gaussians.exposure_optimizer.zero_grad(set_to_none = True)
            if use_sparse_adam:
                visible = radii > 0
                gaussians.optimizer.step(visible, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none = True)
            else:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

        optim_end.record()  # 新增，优化器步骤后
        torch.cuda.synchronize()  # 新增，确保事件记录同步
        iter_time = iter_start.elapsed_time(iter_end)  # 新增
        optim_time = optim_start.elapsed_time(optim_end)  # 新增
        total_time += (iter_time + optim_time) / 1e3  # 新增，单位为秒

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            
            # 更新进度条显示
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
                
                # wandb日志（每10次迭代记录一次）
                if WANDB_FOUND and wandb.run is not None:
                    wandb_log_dict = {
                        'train/ema_loss': ema_loss_for_log,
                        'train/n_gaussians': gaussians.get_xyz.shape[0],
                        'iteration': iteration
                    }
                    if kernel_time is not None:
                        wandb_log_dict['train/kernel_time'] = kernel_time
                    wandb.log(wandb_log_dict)
            
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., None, dataset.train_test_exp, SPARSE_ADAM_AVAILABLE), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    # 训练结束后写入TRAIN_INFO文件
    with open(os.path.join(scene.model_path, "TRAIN_INFO"), "w+") as f:
        f.write("Training Time: {:.2f} seconds, {:.2f} minutes\n".format(total_time, total_time / 60.))
        f.write("GS Number: {}\n".format(gaussians.get_xyz.shape[0]))
        f.write("Total Kernel Time: {:.2f} seconds, {:.2f} minutes\n".format(total_kernel_time, total_kernel_time / 60.))
    # 训练结束后log到wandb
    if WANDB_FOUND and wandb.run is not None:
        wandb.log({
            'final/training_time_seconds': total_time,
            'final/training_time_minutes': total_time / 60.0,
            'final/total_gaussians': gaussians.get_xyz.shape[0],
            'final/total_kernel_time_seconds': total_kernel_time,
            'final/total_kernel_time_minutes': total_kernel_time / 60.0
        })
        wandb.finish()

def prepare_output_and_logger(args, wandb_args=None):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    # 初始化wandb
    if wandb_args and hasattr(wandb_args, 'use_wandb') and wandb_args.use_wandb and WANDB_FOUND:
        wandb.init(
            project=wandb_args.wandb_project if hasattr(wandb_args, 'wandb_project') else "orggs",
            name=wandb_args.wandb_name if hasattr(wandb_args, 'wandb_name') and wandb_args.wandb_name else os.path.basename(args.model_path),
            config=vars(args),
            dir=args.model_path
        )
        print(f"Wandb initialized with project: {wandb_args.wandb_project if hasattr(wandb_args, 'wandb_project') else 'orggs'}")
    elif wandb_args and hasattr(wandb_args, 'use_wandb') and wandb_args.use_wandb and not WANDB_FOUND:
        print("Wandb requested but not available: not logging to wandb")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    # wandb日志
    if WANDB_FOUND and wandb.run is not None:
        wandb.log({
            'train_loss_patches/l1_loss': Ll1.item(),
            'train_loss_patches/total_loss': loss.item(),
            'iter_time': elapsed,
            'iteration': iteration
        })
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})


        total_render_time = 0.0
        total_frames = 0
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_start = torch.cuda.Event(enable_timing = True)
                    render_end = torch.cuda.Event(enable_timing = True)
                    render_start.record()
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    render_end.record()
                    torch.cuda.synchronize()
                    render_time = render_start.elapsed_time(render_end)  # 单位：毫秒
                    total_render_time += render_time / 1000.0  # 转换为秒
                    total_frames += 1

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    # 计算各种指标
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    
                    # 计算SSIM
                    if FUSED_SSIM_AVAILABLE:
                        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                    else:
                        ssim_value = ssim(image, gt_image)
                    ssim_test += ssim_value.double()
                    
                    # 计算LPIPS
                    if LPIPS_AVAILABLE:
                        lpips_value = lpips(image.unsqueeze(0), gt_image.unsqueeze(0), net_type='vgg')
                        lpips_test += lpips_value.double()
                
                # 计算平均值
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                if LPIPS_AVAILABLE:
                    lpips_test /= len(config['cameras'])
                
                # 打印结果
                if LPIPS_AVAILABLE:
                    print("\n[ITER {}] Evaluating {}: L1 {:.6f} PSNR {:.6f} SSIM {:.6f} LPIPS {:.6f}".format(
                        iteration, config['name'], l1_test.item(), psnr_test.item(), ssim_test.item(), lpips_test.item()))
                else:
                    print("\n[ITER {}] Evaluating {}: L1 {:.6f} PSNR {:.6f} SSIM {:.6f}".format(
                        iteration, config['name'], l1_test.item(), psnr_test.item(), ssim_test.item()))
                
                # TensorBoard日志
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    if LPIPS_AVAILABLE:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                
                # WandB日志
                if WANDB_FOUND and wandb.run is not None:
                    wandb_log_dict = {
                        f'{config["name"]}/loss_viewpoint - l1_loss': l1_test.item(),
                        f'{config["name"]}/loss_viewpoint - psnr': psnr_test.item(),
                        f'{config["name"]}/loss_viewpoint - ssim': ssim_test.item(),
                        'iteration': iteration
                    }
                    if LPIPS_AVAILABLE:
                        wandb_log_dict[f'{config["name"]}/loss_viewpoint - lpips'] = lpips_test.item()
                    wandb.log(wandb_log_dict)
        # 计算平均FPS
        if total_frames > 0:
            mean_fps = total_frames / total_render_time
            print(f"[ITER {iteration}] Average FPS: {mean_fps:.2f} (Total frames: {total_frames}, Total time: {total_render_time:.3f}s)")
            if tb_writer:
                tb_writer.add_scalar('evaluation/mean_fps', mean_fps, iteration)
            if WANDB_FOUND and wandb.run is not None:
                wandb.log({
                    'evaluation/mean_fps': mean_fps,
                    'evaluation/total_frames': total_frames,
                    'evaluation/total_render_time': total_render_time,
                    'iteration': iteration
                })
        
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script参数")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000 * (i + 1) for i in range(30)])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10000, 20000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000, 20000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # wandb参数
    parser.add_argument("--use_wandb", action="store_true", default=False, help="启用wandb日志")
    parser.add_argument("--wandb_project", type=str, default="orggs", help="wandb项目名")
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb运行名（可选）")
    # 稀疏Adam参数
    parser.add_argument("--sparse_adam", action="store_true", default=False, help="使用稀疏 Adam 优化器")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # 设置优化器类型
    if args.sparse_adam:
        args.optimizer_type = "sparse_adam"
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
    # All done
    print("\nTraining complete.")