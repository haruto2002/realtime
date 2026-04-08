import cv2
import numpy as np
import os
import torch
from utils.util_dnn import suggest_network
import click
from omegaconf import OmegaConf
from dataset.dataset_utils import copy_datasets, suggest_dataset_root_dir
from tqdm import tqdm
import glob
import pandas as pd
from scipy.spatial import distance_matrix

from metric.utils_predict import load_from_path, predict_slide, predict


def calc_loc_metrics(dist_matrix, gt_num, det_num, sigma):
    if dist_matrix is None:
        if det_num == 0:
            tp, fp, fn = (0, 0, gt_num)
            indices = None
            return tp, fp, fn, indices
        elif gt_num == 0:
            tp, fp, fn = (0, det_num, 0)
            indices = None
            return tp, fp, fn, indices

    match_matrix = np.zeros(dist_matrix.shape, dtype=bool)
    pred_num = dist_matrix.shape[0]
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p, :]
        match_matrix[i_pred_p, :] = pred_dist <= sigma

    tp, assign = hungarian(match_matrix)
    fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
    tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
    tp_gt_index = np.array(np.where(assign.sum(0) == 1))[0]
    fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]

    indices = (tp_gt_index, fn_gt_index, tp_pred_index, fp_pred_index)

    tp = tp_pred_index.shape[0]
    fp = fp_pred_index.shape[0]
    fn = fn_gt_index.shape[0]

    return tp, fp, fn, indices


#  Hungarian method for bipartite graph
def hungarian(matrixTF):
    # matrix to adjacent matrix
    edges = np.argwhere(matrixTF)
    lnum, rnum = matrixTF.shape
    graph = [[] for _ in range(lnum)]
    for edge in edges:
        graph[edge[0]].append(edge[1])

    # deep first search
    match = [-1 for _ in range(rnum)]
    vis = [-1 for _ in range(rnum)]

    def dfs(u):
        for v in graph[u]:
            if vis[v]:
                continue
            vis[v] = True
            if match[v] == -1 or dfs(match[v]):
                match[v] = u
                return True
        return False

    # for loop
    ans = 0
    for a in range(lnum):
        for i in range(rnum):
            vis[i] = False
        if dfs(a):
            ans += 1

    # assignment matrix
    assign = np.zeros((lnum, rnum), dtype=bool)
    for i, m in enumerate(match):
        if m >= 0:
            assign[m, i] = True

    return ans, assign


def calc_dist(detections, gts):
    if len(detections) == 0 or len(gts) == 0:
        return None
    pred_points = np.array([detection["coordinate"] for detection in detections])
    gt_points = np.array([gt["coordinate"] for gt in gts])
    dist_matrix = distance_matrix(pred_points, gt_points)
    return dist_matrix


def calc_cnt_metric(detections, gts):
    ae = abs(len(detections) - len(gts))
    se = (len(detections) - len(gts)) ** 2
    gt_cnt, pred_cnt = len(gts), len(detections)
    return (ae, se, gt_cnt, pred_cnt)


def each_saver(
    save_dir,
    dataset_name,
    img_id,
    weight_name,
    sigma,
    tp,
    fp,
    fn,
    precision,
    recall,
    f1,
    ae,
    se,
    gt_cnt,
    pred_cnt,
):

    text = (
        f"Dataset:{dataset_name} \n"
        + f"Image ID:{img_id} \n"
        + f"Weight name:{weight_name} \n"
        + "\nLocalization Metric \n"
        + f"Sigma:{sigma} \n"
        + f"Presicion:{precision} \n"
        + f"Recall:{recall} \n"
        + f"F1 Score:{f1} \n"
        + f"TP:{tp} \n"
        + f"FP:{fp} \n"
        + f"FN:{fn} \n"
        + "\nCounting Metric \n"
        + f"AE:{ae} \n"
        + f"SE:{se} \n"
        + f"GT:{gt_cnt} \n"
        + f"Prediction:{pred_cnt} \n"
    )
    with open(f"{save_dir}/metric_score_sigma{sigma}.txt", "w") as file:
        file.write(text)


def total_saver(
    save_dir,
    dataset_name,
    weight_name,
    sigma,
    total_tp,
    total_fp,
    total_fn,
    total_precision,
    total_recall,
    total_f1,
    total_ae,
    total_se,
):
    columns = ["TP", "FP", "FN", "Precision", "Recall", "F1", "AE", "SE"]
    results = [
        [
            total_tp[i],
            total_fp[i],
            total_fn[i],
            total_precision[i],
            total_recall[i],
            total_f1[i],
            total_ae[i],
            total_se[i],
        ]
        for i in range(len(total_tp))
    ]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(f"{save_dir}/results_sigma{sigma}.csv", index=False)

    tp = np.sum(total_tp)
    fp = np.sum(total_fp)
    fn = np.sum(total_fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    mae = np.mean(total_ae)
    rmse = np.sqrt(np.mean(total_se))

    text = (
        f"Dataset:{dataset_name} \n"
        + f"Weight name:{weight_name} \n"
        + "\nLocalization Metric \n"
        + f"Sigma:{sigma} \n"
        + f"Presicion:{precision} \n"
        + f"Recall:{recall} \n"
        + f"F1 Score:{f1} \n"
        + f"TP:{tp} \n"
        + f"FP:{fp} \n"
        + f"FN:{fn} \n"
        + "\nCounting Metric \n"
        + f"MAE:{mae} \n"
        + f"RMSE:{rmse} \n"
    )
    with open(f"{save_dir}/total_metric_score_sigma{sigma}.txt", "w") as file:
        file.write(text)


def vis_point(save_dir, img, gts, detections, indices, sigma):
    if indices is None:
        return
    tp_gt_index, fn_gt_index, tp_pred_index, fp_pred_index = indices
    color_list = [(0, 255, 0), (255, 255, 0), (0, 0, 255), (0, 255, 255)]
    size_list = [sigma, sigma, 2, 2]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_copy = img.copy()
    for i, idx_list in enumerate(indices):
        for idx in idx_list:
            if i <= 1:
                p = gts[idx]["coordinate"]
                thickness = 1

            else:
                p = detections[idx]["coordinate"]
                thickness = -1

            cv2.circle(
                img_copy,
                (int(p[0]), int(p[1])),
                size_list[i],
                color_list[i],
                thickness,
                lineType=cv2.LINE_AA,
            )
    cv2.imwrite(
        save_dir
        + f"/sigma{sigma}_match{len(tp_gt_index)}_unmatchgt{len(fn_gt_index)}_misspred{len(fp_pred_index)}.png",
        img_copy,
    )


def vis_point2(save_dir, path2img, gts, detections, indices, sigma):
    # if indices is None:
    # add code

    tp_gt_index, fn_gt_index, tp_pred_index, fp_pred_index = indices
    img = cv2.imread(path2img)
    img_copy = img.copy()
    for i, idx_list in enumerate(indices[1:]):
        # 0:FN(gt) >> 1:TP(pred) >> 2:FP(pred)
        for idx in idx_list:
            if i == 0:  # FN
                p = gts[idx]["coordinate"]
                color = (255, 0, 0)

            else:
                p = detections[idx]["coordinate"]
                if i == 1:  # TP
                    color = (0, 255, 0)
                elif i == 2:  # FP
                    color = (0, 0, 255)

            cv2.circle(
                img_copy,
                (int(p[0]), int(p[1])),
                5,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )
    cv2.imwrite(
        save_dir
        + f"/sigma{sigma}_TP{len(tp_gt_index)}_FN{len(fn_gt_index)}_FP{len(fp_pred_index)}.png",
        img_copy,
    )


def calc_f1_score(tp, fp, fn):
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def calc_metric(
    save_dir,
    dataset_name,
    weight_name,
    data_dir_list,
    model,
    device,
    sigma=4,
    each_save=True,
):
    # total_tp, total_fp, total_fn, total_ae, total_se = 0, 0, 0, 0, 0
    total_tp = np.zeros(len(data_dir_list))
    total_fp = np.zeros(len(data_dir_list))
    total_fn = np.zeros(len(data_dir_list))
    total_precision = np.zeros(len(data_dir_list))
    total_recall = np.zeros(len(data_dir_list))
    total_f1 = np.zeros(len(data_dir_list))
    total_ae = np.zeros(len(data_dir_list))
    total_se = np.zeros(len(data_dir_list))
    for i, data_dir in enumerate(tqdm(data_dir_list)):
        path2img = glob.glob(data_dir + "/*.jpg")[0]
        path2gt = glob.glob(data_dir + "/*.txt")[0]
        dir_name = data_dir.split("/")[-1]
        img_id = int(dir_name[5:])  # Extract number from scene0001
        each_save_dir = save_dir + "/each_results/" + dir_name

        img, gt_points = load_from_path(path2img, path2gt)
        if img.shape == (4320, 7680, 3) or img.shape == (4320, 3840, 3):
            # print("slide")
            h_size = 720
            w_size = 1280

            # h_size = 1080
            # w_size = 1920

            # h_size = 128
            # w_size = 128

            # h_size = 2160
            # w_size = 3840
            detections, gts, _ = predict_slide(
                img_id, img, gt_points, model, device, h_size=h_size, w_size=w_size
            )
        else:
            detections, gts, transformed_img = predict(
                dataset_name, img_id, img, gt_points, model, device
            )

        gt_num = len(gts)
        det_num = len(detections)
        dist_matrix = calc_dist(detections, gts)
        tp, fp, fn, indices = calc_loc_metrics(dist_matrix, gt_num, det_num, sigma)
        precision, recall, f1 = calc_f1_score(tp, fp, fn)
        ae, se, gt_cnt, pred_cnt = calc_cnt_metric(detections, gts)

        if each_save:
            os.makedirs(each_save_dir, exist_ok=True)
            vis_point(each_save_dir, transformed_img, gts, detections, indices, sigma)
            # vis_point2(each_save_dir, path2img, gts, detections, indices, sigma)
            each_saver(
                each_save_dir,
                dataset_name,
                img_id,
                weight_name,
                sigma,
                tp,
                fp,
                fn,
                precision,
                recall,
                f1,
                ae,
                se,
                gt_cnt,
                pred_cnt,
            )
        total_tp[i] = tp
        total_fp[i] = fp
        total_fn[i] = fn
        total_precision[i] = precision
        total_recall[i] = recall
        total_f1[i] = f1
        total_ae[i] = ae
        total_se[i] = se

    total_saver(
        save_dir,
        dataset_name,
        weight_name,
        sigma,
        total_tp,
        total_fp,
        total_fn,
        total_precision,
        total_recall,
        total_f1,
        total_ae,
        total_se,
    )


@click.command()
@click.argument("save_dir", type=str, default="METRIC_demo")
@click.argument(
    "weight_path",
    type=str,
    default="SHTechA.pth",
)
@click.argument("weight_name", type=str, default="SHTechA")
@click.argument("dataset_name", type=str, default="full_WorldPorters2023")
@click.argument("task", type=str, default="counting")
@click.argument("resource", type=str, default="local")
@click.argument("sigma", type=int, default=12)
@click.argument("each_save", type=bool, default=True)
@click.argument("gpu_id", type=str, default=0)
def main(
    save_dir,
    weight_path,
    weight_name,
    dataset_name,
    task,
    resource,
    sigma,
    each_save,
    gpu_id,
):
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"

    save_dir = f"{save_dir}/{dataset_name}/{weight_name}"
    os.makedirs(save_dir, exist_ok=True)

    cfg_path = "p2pnet/conf/p2p.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.dataset.name = dataset_name
    cfg.default.resource = resource
    cfg.network.dnn_task = task
    # cfg.network.dnn_task = "hanabi"
    cfg.default.finetune = True
    cfg.network.init_weight = weight_path
    cfg.network.row = 4

    model = suggest_network(cfg, device)
    model.to(device)
    model.eval()

    copy_datasets(cfg)
    root_dir = suggest_dataset_root_dir(cfg)

    if (
        dataset_name == "full_WorldPorters2023"
        or dataset_name == "full_WorldPorters2023_large"
        or dataset_name == "full_WorldPorters2023_all"
        or dataset_name == "testset"
    ):
        data_dir_list = sorted(glob.glob(root_dir + "/*.jpg"))
    else:
        data_dir_list = sorted(glob.glob(root_dir + "test/*"))
        # data_dir_list = sorted(glob.glob(root_dir + "test/*"))[:2]  # debug

    # print(data_dir_list)
    calc_metric(
        save_dir,
        dataset_name,
        weight_name,
        data_dir_list,
        model,
        device,
        sigma=sigma,
        each_save=each_save,
    )


if __name__ == "__main__":
    main()
