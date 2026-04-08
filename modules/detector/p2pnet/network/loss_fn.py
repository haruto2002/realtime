import torch
import torch.nn.functional as F
from torch import nn


class SetCriterion_Crowd(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # outputs: dict{'pred_logits': tensor([probability of person, probability of not person]...per coordinate)[32, 1024, 2],'pred_points':...(not used)}
        # targets:各画像に対す辞書(['point', 'image_id', 'labels'])
        # indices:各画像の(preのidxの順列,gtのidxの順列) (32,2)

        assert "pred_logits" in outputs
        p2pnet_logits = outputs[
            "pred_logits"
        ]  # (32,1024(画像内のreference pointsの数：128/8x128/8x4),2)

        idx = self._get_p2pnet_permutation_idx(
            indices
        )  # (対応するbatch_numのテンソル、preのidxのテンソル)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )  # バッチ内の全gt個分の１のテンソル
        target_classes = torch.full(
            p2pnet_logits.shape[:2], 0, dtype=torch.int64, device=p2pnet_logits.device
        )  # (32,1024)のゼロテンソル（バッチごとの全参照点分）
        target_classes[idx] = target_classes_o  # gtと対応しているものだけ１に変更

        # print(p2pnet_logits.transpose(1, 2).device) >> cuda
        # print(target_classes.device) >> cuda
        # print(self.empty_weight.device) >> cpu
        # Error occurs here if loss_fn is not loaded on GPU
        loss_ce = F.cross_entropy(
            p2pnet_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}

        return losses

    def loss_points(self, outputs, targets, indices, num_points):

        # outputs: dict{'pred_logits':...(not used),'pred_points': coordinates of all 1024 reference points per batch[32, 1024, 2]}
        # targets:各画像に対す辞書(['point', 'image_id', 'labels'])
        # indices:各画像の(preのidxの順列,gtのidxの順列) (32,2)

        assert "pred_points" in outputs
        idx = self._get_p2pnet_permutation_idx(indices)
        p2pnet_points = outputs["pred_points"][idx]  # gtに対応するpreの座標だけ抜き出す
        target_points = torch.cat(
            [t["point"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )  # gtの座標を抜き出す

        loss_bbox = F.mse_loss(p2pnet_points, target_points, reduction="none")

        losses = {}
        losses["loss_points"] = loss_bbox.sum() / num_points
        # losses["loss_points"] = torch.nan_to_num(losses["loss_points"], nan=0.0)
        # print(losses)
        return losses

    def _get_p2pnet_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(p2pnet, i) for i, (p2pnet, _) in enumerate(indices)]
        )
        p2pnet_idx = torch.cat([p2pnet for (p2pnet, _) in indices])
        return batch_idx, p2pnet_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            "labels": self.loss_labels,  # This is a function, not a variable
            "points": self.loss_points,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {
            "pred_logits": outputs["pred_logits"],
            "pred_points": outputs["pred_points"],
        }

        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor(
            [num_points], dtype=torch.float, device=next(iter(output1.values())).device
        )

        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_points)
        # num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            # losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))
            losses.update(self.get_loss(loss, output1, targets, indices1, num_points))

        return losses
