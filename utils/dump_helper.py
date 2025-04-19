# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import os
import sys
import trimesh

DUMP_CONF_THRESH = 0.5 # Dump boxes with obj prob larger than that.


def create_lineset_from_corners(corners, color=[1, 0, 0]):
    """从8个角点创建线框"""
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底部矩形
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶部矩形
        [0, 4], [1, 5], [2, 6], [3, 7]   # 连接顶部和底部的线
    ]
    
    # 创建线框的顶点和边
    vertices = corners
    edges = np.array(lines)
    
    # 返回顶点和边
    return vertices, edges

def save_bbox_as_ply(vertices, edges, filepath, color=[1, 0, 0]):
    """将边界框保存为PLY文件"""
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")
        
        # 写入顶点
        r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]} {r} {g} {b}\n")
        
        # 写入边
        for e in edges:
            f.write(f"{e[0]} {e[1]}\n")

def visualize_scene(point_cloud, pred_boxes, gt_boxes, pred_scores=None, pred_sem_labels=None, 
                   gt_sem_labels=None, max_boxes=256, pc_id=0, save_dir=None):
    # 保存点云到文件
    if save_dir is not None:
        pc_path = os.path.join(save_dir, f'{pc_id:06d}_pc.ply')
        # 使用numpy保存点云为PLY文件
        with open(pc_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(point_cloud)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in point_cloud:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        # print(f"点云已保存到 {pc_path}")

    # 保存预测的边界框
    if pred_boxes is not None and pred_sem_labels is not None and save_dir is not None:
        valid_pred_boxes = []
        for i in range(min(len(pred_boxes), max_boxes)):
            if pred_scores is not None and pred_scores[i]  < DUMP_CONF_THRESH:
                continue
            # 获取最大的预测类别
            sem_cls = np.argmax(pred_sem_labels[i])
            # print(f"pred_sem_cls: {sem_cls}")
            valid_pred_boxes.append(pred_boxes[i])
        
        if valid_pred_boxes:
            pred_path = os.path.join(save_dir, f'{pc_id:06d}_pred_bbox.ply')
            
            # 合并所有有效的边界框
            all_vertices = []
            all_edges = []
            vertex_offset = 0
            
            for box in valid_pred_boxes:
                vertices, edges = create_lineset_from_corners(box, color=[1, 0, 0])
                all_vertices.append(vertices)
                # 调整边的索引
                adjusted_edges = edges + vertex_offset
                all_edges.append(adjusted_edges)
                vertex_offset += len(vertices)
            
            # 合并顶点和边
            all_vertices = np.vstack(all_vertices)
            all_edges = np.vstack(all_edges)
            
            # 保存为PLY文件
            save_bbox_as_ply(all_vertices, all_edges, pred_path, color=[1, 0, 0])
            
            # print(f"预测边界框已保存到 {pred_path}")

    # 保存真实的边界框
    if gt_boxes is not None and save_dir is not None:
        valid_gt_boxes = []
        for i in range(min(len(gt_boxes), max_boxes)):
            if np.sum(gt_boxes[i]) == 0:
                continue
            # print(f"gt_sem_cls: {gt_sem_labels[i]}")
            valid_gt_boxes.append(gt_boxes[i])
        
        if valid_gt_boxes:
            gt_path = os.path.join(save_dir, f'{pc_id:06d}_gt_bbox.ply')
            
            # 合并所有有效的边界框
            all_vertices = []
            all_edges = []
            vertex_offset = 0
            
            for box in valid_gt_boxes:
                vertices, edges = create_lineset_from_corners(box, color=[0, 1, 0])
                all_vertices.append(vertices)
                # 调整边的索引
                adjusted_edges = edges + vertex_offset
                all_edges.append(adjusted_edges)
                vertex_offset += len(vertices)
            
            # 合并顶点和边
            all_vertices = np.vstack(all_vertices)
            all_edges = np.vstack(all_edges)
            
            # 保存为PLY文件
            save_bbox_as_ply(all_vertices, all_edges, gt_path, color=[0, 1, 0])
            # print(f"真实边界框已保存到 {gt_path}")


def dump_results(output, gt, dump_dir, batch_idx):
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''
    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = gt['point_clouds'].cpu().numpy() # B,N,4
    batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    pred_objectness_scores = output['objectness_prob'].cpu().numpy() # B,K
    pred_sem_cls_probs = output['sem_cls_prob'].cpu().numpy() # B,K,53
    pred_bboxs = output['box_corners'].cpu().numpy() # B,K,8,3

    # Ground truth
    gt_box_sem_cls_label = gt['gt_box_sem_cls_label'].cpu().numpy() # B,K
    gt_bboxs = gt['gt_box_corners'].cpu().numpy() # B,K,8,3

    # id
    name_id = batch_idx * batch_size

    for i in range(batch_size):
        idx = name_id + i
        pc = point_clouds[i,:,:]
        pc[:, :] = pc[:, [0,2,1]] # x,z,y to x,y,z
        pc[..., 1] *= -1
        visualize_scene(pc, pred_bboxs[i], gt_bboxs[i], pred_scores=pred_objectness_scores[i], 
                        pred_sem_labels=pred_sem_cls_probs[i], gt_sem_labels=gt_box_sem_cls_label[i], 
                        pc_id=idx, save_dir=dump_dir)

        # # dump pc
        # pc_util.write_ply(pc, os.path.join(dump_dir, '%06d_pc.ply'%(idx)))

        # # dump GT bounding boxes
        # gt_bboxs = gt_bboxs[i,gt_box_sem_cls_label[i,:]>0,:,:].reshape(-1,3)
        # print(gt_bboxs.shape)


        # # Dump predicted bounding boxes
        # if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
        #     pred_bbox = pred_bboxs[i,objectness_prob>DUMP_CONF_THRESH,:,:].reshape(-1,3)
        #     print(pred_bbox.shape)
        #     pc_util.write_oriented_bbox(pred_bbox, os.path.join(dump_dir, '%06d_pred_bbox.ply'%(idx)))

    