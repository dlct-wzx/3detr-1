import torch
import numpy as np
import open3d as o3d
import argparse
import os
import matplotlib.pyplot as plt

def create_lineset_from_corners(corners, color=[1, 0, 0]):
    """从8个角点创建线框"""
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底部矩形
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶部矩形
        [0, 4], [1, 5], [2, 6], [3, 7]   # 连接顶部和底部的线
    ]
    
    # 创建线框
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set

def visualize_scene(point_cloud, pred_boxes, gt_boxes, pred_scores=None, pred_sem_labels=None, 
                   gt_sem_labels=None, score_thresh=0.1, max_boxes=50):
    """可视化点云和边界框"""
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # 设置点云颜色为灰色
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Detection Visualization", width=1280, height=720)
    
    # 添加点云
    vis.add_geometry(pcd)
    
    # 添加预测的边界框（红色）
    if pred_boxes is not None:
        for i in range(min(len(pred_boxes), max_boxes)):
            # 如果有置信度分数，则过滤低分数的框
            if pred_scores is not None and pred_scores[i] < score_thresh:
                continue
                
            corners = pred_boxes[i]
            line_set = create_lineset_from_corners(corners, color=[1, 0, 0])  # 红色
            vis.add_geometry(line_set)
            
            # 如果有语义标签，可以在框上方添加标签
            if pred_sem_labels is not None:
                label = pred_sem_labels[i]
                # 这里可以添加标签显示代码，但Open3D不直接支持文本标签
                # 可以考虑使用其他方法如添加小球体等表示不同类别
    
    # 添加真实的边界框（绿色）
    if gt_boxes is not None:
        for i in range(min(len(gt_boxes), max_boxes)):
            corners = gt_boxes[i]
            line_set = create_lineset_from_corners(corners, color=[0, 1, 0])  # 绿色
            vis.add_geometry(line_set)
    
    # 设置视图控制
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # 黑色背景
    opt.point_size = 1.0
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

def load_checkpoint_and_visualize(checkpoint_path, batch_idx=0, score_thresh=0.1):
    """加载检查点并可视化"""
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # 提取数据
    outputs = checkpoint.get('outputs', {})
    batch_data = checkpoint.get('batch_data', {})
    
    # 获取点云数据
    point_clouds = batch_data.get('point_clouds', None)
    if point_clouds is None:
        print("未找到点云数据")
        return
    
    point_cloud = point_clouds[batch_idx].cpu().numpy()
    
    # 获取预测的边界框
    pred_box_corners = outputs.get('box_corners', None)
    if pred_box_corners is not None:
        pred_box_corners = pred_box_corners[batch_idx].cpu().numpy()
    
    # 获取预测的置信度分数
    pred_scores = outputs.get('objectness_prob', None)
    if pred_scores is not None:
        pred_scores = pred_scores[batch_idx].cpu().numpy()
    
    # 获取预测的语义标签
    pred_sem_labels = outputs.get('sem_cls_prob', None)
    if pred_sem_labels is not None:
        pred_sem_labels = torch.argmax(pred_sem_labels[batch_idx], dim=1).cpu().numpy()
    
    # 获取真实的边界框
    gt_box_corners = batch_data.get('gt_box_corners', None)
    if gt_box_corners is not None:
        gt_box_corners = gt_box_corners[batch_idx].cpu().numpy()
        # 过滤掉无效的边界框（使用gt_box_present）
        gt_box_present = batch_data.get('gt_box_present', None)
        if gt_box_present is not None:
            gt_box_present = gt_box_present[batch_idx].cpu().numpy()
            gt_box_corners = gt_box_corners[gt_box_present > 0.5]
    
    # 获取真实的语义标签
    gt_sem_labels = batch_data.get('gt_box_sem_cls_label', None)
    if gt_sem_labels is not None:
        gt_sem_labels = gt_sem_labels[batch_idx].cpu().numpy()
        # 同样过滤无效的标签
        if gt_box_present is not None:
            gt_sem_labels = gt_sem_labels[gt_box_present > 0.5]
    
    # 可视化
    visualize_scene(
        point_cloud=point_cloud,
        pred_boxes=pred_box_corners,
        gt_boxes=gt_box_corners,
        pred_scores=pred_scores,
        pred_sem_labels=pred_sem_labels,
        gt_sem_labels=gt_sem_labels,
        score_thresh=score_thresh
    )

def visualize_from_tensors(outputs, batch_data, batch_idx=0, score_thresh=0.1):
    """直接从模型输出的张量进行可视化"""
    # 获取点云数据
    point_clouds = batch_data.get('point_clouds', None)
    if point_clouds is None:
        print("未找到点云数据")
        return
    
    point_cloud = point_clouds[batch_idx].cpu().numpy()
    
    # 获取预测的边界框
    pred_box_corners = outputs.get('box_corners', None)
    if pred_box_corners is not None:
        pred_box_corners = pred_box_corners[batch_idx].cpu().numpy()
    
    # 获取预测的置信度分数
    pred_scores = outputs.get('objectness_prob', None)
    if pred_scores is not None:
        pred_scores = pred_scores[batch_idx].cpu().numpy()
    
    # 获取预测的语义标签
    pred_sem_labels = outputs.get('sem_cls_prob', None)
    if pred_sem_labels is not None:
        pred_sem_labels = torch.argmax(pred_sem_labels[batch_idx], dim=1).cpu().numpy()
    
    # 获取真实的边界框
    gt_box_corners = batch_data.get('gt_box_corners', None)
    if gt_box_corners is not None:
        gt_box_corners = gt_box_corners[batch_idx].cpu().numpy()
        # 过滤掉无效的边界框（使用gt_box_present）
        gt_box_present = batch_data.get('gt_box_present', None)
        if gt_box_present is not None:
            gt_box_present = gt_box_present[batch_idx].cpu().numpy()
            gt_box_corners = gt_box_corners[gt_box_present > 0.5]
    
    # 获取真实的语义标签
    gt_sem_labels = batch_data.get('gt_box_sem_cls_label', None)
    if gt_sem_labels is not None:
        gt_sem_labels = gt_sem_labels[batch_idx].cpu().numpy()
        # 同样过滤无效的标签
        if gt_box_present is not None:
            gt_sem_labels = gt_sem_labels[gt_box_present > 0.5]
    
    # 可视化
    visualize_scene(
        point_cloud=point_cloud,
        pred_boxes=pred_box_corners,
        gt_boxes=gt_box_corners,
        pred_scores=pred_scores,
        pred_sem_labels=pred_sem_labels,
        gt_sem_labels=gt_sem_labels,
        score_thresh=score_thresh
    )

def save_visualization_to_file(outputs, batch_data, save_path, batch_idx=0, score_thresh=0.1):
    """保存可视化结果到文件"""
    # 获取点云数据
    point_clouds = batch_data.get('point_clouds', None)
    if point_clouds is None:
        print("未找到点云数据")
        return
    
    point_cloud = point_clouds[batch_idx].cpu().numpy()
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    
    # 创建可视化列表
    geometries = [pcd]
    
    # 添加预测的边界框（红色）
    pred_box_corners = outputs.get('box_corners', None)
    pred_scores = outputs.get('objectness_prob', None)
    
    if pred_box_corners is not None:
        pred_box_corners = pred_box_corners[batch_idx].cpu().numpy()
        if pred_scores is not None:
            pred_scores = pred_scores[batch_idx].cpu().numpy()
        
        for i in range(len(pred_box_corners)):
            if pred_scores is not None and pred_scores[i] < score_thresh:
                continue
            corners = pred_box_corners[i]
            line_set = create_lineset_from_corners(corners, color=[1, 0, 0])
            geometries.append(line_set)
    
    # 添加真实的边界框（绿色）
    gt_box_corners = batch_data.get('gt_box_corners', None)
    if gt_box_corners is not None:
        gt_box_corners = gt_box_corners[batch_idx].cpu().numpy()
        gt_box_present = batch_data.get('gt_box_present', None)
        
        if gt_box_present is not None:
            gt_box_present = gt_box_present[batch_idx].cpu().numpy()
            valid_gt_boxes = gt_box_corners[gt_box_present > 0.5]
            
            for corners in valid_gt_boxes:
                line_set = create_lineset_from_corners(corners, color=[0, 1, 0])
                geometries.append(line_set)
    
    # 保存可视化结果
    o3d.visualization.draw_geometries_with_custom_animation(
        geometries,
        window_name="3D Detection Visualization",
        width=1280,
        height=720,
        left=50,
        top=50,
        optional_view_trajectory_json_file=save_path
    )
    
    # print(f"可视化结果已保存到 {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D边界框可视化工具")
    parser.add_argument("--checkpoint", type=str, help="检查点文件路径")
    parser.add_argument("--batch_idx", type=int, default=0, help="要可视化的批次索引")
    parser.add_argument("--score_thresh", type=float, default=0.1, help="置信度阈值")
    parser.add_argument("--save_path", type=str, default=None, help="保存可视化结果的路径")
    
    args = parser.parse_args()
    
    if args.checkpoint:
        load_checkpoint_and_visualize(args.checkpoint, args.batch_idx, args.score_thresh)
    else:
        print("请提供检查点文件路径")