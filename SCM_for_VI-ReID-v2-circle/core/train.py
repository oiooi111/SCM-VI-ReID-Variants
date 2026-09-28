import torch
import torch.nn.functional as F
from tools import MultiItemAverageMeter
from network.processing import FeatureShuffling


def _shared_identity_means(features_a, labels_a, features_b, labels_b):
    """Return modality-specific feature/logit means for identities in both inputs."""
    means_a = []
    means_b = []
    for identity in torch.unique(labels_a):
        mask_b = labels_b == identity
        if not torch.any(mask_b):
            continue
        means_a.append(features_a[labels_a == identity].mean(dim=0))
        means_b.append(features_b[mask_b].mean(dim=0))

    if not means_a:
        return None, None
    return torch.stack(means_a, dim=0), torch.stack(means_b, dim=0)


def symmetric_kl_loss(logits_a, logits_b, temperature=2.0):
    """Symmetric KL divergence with temperature scaling."""
    log_prob_a = F.log_softmax(logits_a.float() / temperature, dim=1)
    log_prob_b = F.log_softmax(logits_b.float() / temperature, dim=1)
    prob_a = log_prob_a.exp()
    prob_b = log_prob_b.exp()
    kl_ab = F.kl_div(log_prob_a, prob_b, reduction='batchmean')
    kl_ba = F.kl_div(log_prob_b, prob_a, reduction='batchmean')
    return 0.5 * (kl_ab + kl_ba) * (temperature ** 2)


def cross_modal_center_loss(features_a, labels_a, features_b, labels_b):
    """Align normalized per-identity centers instead of arbitrary image pairs."""
    centers_a, centers_b = _shared_identity_means(features_a, labels_a, features_b, labels_b)
    if centers_a is None:
        return features_a.sum() * 0.0
    centers_a = F.normalize(centers_a.float(), p=2, dim=1)
    centers_b = F.normalize(centers_b.float(), p=2, dim=1)
    return (1.0 - (centers_a * centers_b).sum(dim=1)).mean()


def train_stage1(base, num_image, i_ter, batch, visible_labels_list, visible_image_features_list,
                  infrared_labels_list, infrared_image_features_list):
    base.set_train()
    meter = MultiItemAverageMeter()
    iter_list = torch.randperm(num_image).to(base.device)
    for i in range(i_ter):
        b_list = iter_list[i*batch: (i+1)*batch]
        rgb_target = visible_labels_list[b_list].long()
        ir_target = infrared_labels_list[b_list].long()
        rgb_image_features = visible_image_features_list[b_list]
        ir_image_features = infrared_image_features_list[b_list]

        # ----------------------------------------------------------------------
        # Instance-specific token (design-1): map image features -> CLIP token
        # ----------------------------------------------------------------------
        # (B, 1024) -> (B, 512)
        inst_v = base.model.module.img2token_v(rgb_image_features)
        inst_r = base.model.module.img2token_r(ir_image_features)

        # Instance-aware text features
        rgb_text_features = base.model(label1=rgb_target, inst1=inst_v, get_text=True)
        ir_text_features = base.model(label2=ir_target, inst2=inst_r, get_text=True)
        image_features = torch.cat([rgb_image_features, ir_image_features], dim=0)
        text_features = torch.cat([rgb_text_features, ir_text_features], dim=0)
        target = torch.cat([rgb_target, ir_target], dim=0)
        loss_i2t = base.con_creiteron(image_features, text_features, target, target)
        loss_t2i = base.con_creiteron(text_features, image_features, target, target)

        # Cross-modal instance token alignment (same identity across modalities)
        inst_v_n = F.normalize(inst_v.float(), dim=1)
        inst_r_n = F.normalize(inst_r.float(), dim=1)
        loss_align = (1.0 - (inst_v_n * inst_r_n).sum(dim=1)).mean()

        loss = loss_i2t + loss_t2i + base.config.lambda_align * loss_align
        base.model_optimizer_stage1.zero_grad()
        loss.backward()
        base.model_optimizer_stage1.step()

        meter.update({'loss_i2t': loss_i2t.data,
                      'loss_t2i': loss_t2i.data,})

        meter.update({'loss_align': loss_align.data})

    return meter.get_val(), meter.get_str()

def train_stage2(base, num_image, i_ter, batch, labels_list, image_features_list):
    base.set_train()
    meter = MultiItemAverageMeter()
    iter_list = torch.randperm(num_image).to(base.device)
    for i in range(i_ter):
        b_list = iter_list[i*batch: (i+1)*batch]
        target = labels_list[b_list].long()
        image_features = image_features_list[b_list]
        text_features = base.model(label=target, get_fusion_text=True)
        loss_i2t = base.con_creiteron(image_features, text_features, target, target)
        loss_t2i = base.con_creiteron(text_features, image_features, target, target)

        loss = loss_i2t + loss_t2i
        base.model_optimizer_stage2.zero_grad()
        loss.backward()
        base.model_optimizer_stage2.step()

        meter.update({'loss_i2t': loss_i2t.data,
                      'loss_t2i': loss_t2i.data,})

    return meter.get_val(), meter.get_str()

def train(base, loaders, text_features, config, current_epoch):

    base.set_train()
    meter = MultiItemAverageMeter()
    loader = loaders.get_train_loader()
    for i, (input1_0, input1_1, input2, label1, label2) in enumerate(loader):
        rgb_imgs1, rgb_imgs2, rgb_pids = input1_0, input1_1, label1
        ir_imgs, ir_pids = input2, label2
        rgb_imgs1, rgb_imgs2, rgb_pids = rgb_imgs1.to(base.device),  rgb_imgs2.to(base.device), \
                                        rgb_pids.to(base.device).long()
        ir_imgs, ir_pids = ir_imgs.to(base.device), ir_pids.to(base.device).long()

        rgb_imgs = torch.cat([rgb_imgs1, rgb_imgs2], dim=0)
        pids = torch.cat([rgb_pids, rgb_pids, ir_pids], dim=0)

        features, cls_score = base.model(x1=rgb_imgs, x2=ir_imgs)

        n = features[1].shape[0] // 3
        rgb_features = features[0].squeeze().narrow(0, 0, n)#32.2048
        ir_features = features[0].squeeze().narrow(0, 2 * n, n)#32,2048
        rgb_attn_features_view1 = features[1].narrow(0, 0, n)
        rgb_attn_features_view2 = features[1].narrow(0, n, n)
        # Average the two independently augmented RGB views for more stable
        # cross-modal supervision. The original per-view features are still
        # optimized by the identity and triplet objectives below.
        rgb_attn_features = 0.5 * (rgb_attn_features_view1 + rgb_attn_features_view2)
        ir_attn_features = features[1].narrow(0, 2 * n, n)

        rgb_logits = rgb_attn_features @ text_features.t()
        ir_logits = ir_attn_features @ text_features.t()

        ide_loss = base.pid_creiteron(cls_score[0], pids)
        ide_loss_proj = base.pid_creiteron(cls_score[1], pids)
        triplet_loss = base.tri_creiteron(features[0].squeeze(), pids)
        triplet_loss_proj = base.tri_creiteron(features[1].squeeze(), pids)
        msel_loss = base.msel_creiteron(torch.cat([rgb_features, ir_features], dim=0), torch.cat([rgb_pids, ir_pids], dim=0))
        msel_loss_proj = base.msel_creiteron(torch.cat([rgb_attn_features, ir_attn_features], dim=0),
                                        torch.cat([rgb_pids, ir_pids], dim=0))

        rgb_i2t_ide_loss = base.pid_creiteron(rgb_logits, rgb_pids)
        ir_i2t_ide_loss = base.pid_creiteron(ir_logits, ir_pids)

        # Causally-motivated modality invariance: compare prediction
        # distributions after averaging samples of the same identity. This is
        # robust to the fact that RGB/IR images are not pose-aligned pairs.
        rgb_cls_logits = 0.5 * (cls_score[1].narrow(0, 0, n) +
                                cls_score[1].narrow(0, n, n))
        ir_cls_logits = cls_score[1].narrow(0, 2 * n, n)
        rgb_cls_centers, ir_cls_centers = _shared_identity_means(
            rgb_cls_logits, rgb_pids, ir_cls_logits, ir_pids)
        if rgb_cls_centers is None:
            inv_loss = rgb_cls_logits.sum() * 0.0
        else:
            inv_loss = symmetric_kl_loss(
                rgb_cls_centers, ir_cls_centers, config.kl_temperature)

        center_loss = cross_modal_center_loss(
            rgb_attn_features, rgb_pids, ir_attn_features, ir_pids)
        circle_loss = base.circle_creiteron(features[1], pids)

        loss = ide_loss + ide_loss_proj + (msel_loss + msel_loss_proj) + \
               config.lambda1 * (triplet_loss + triplet_loss_proj) + \
               config.lambda2 * rgb_i2t_ide_loss + config.lambda3 * ir_i2t_ide_loss + \
               config.lambda_inv * inv_loss + config.lambda_center * center_loss + \
               config.lambda_circle * circle_loss

        base.model_optimizer_stage3.zero_grad()
        loss.backward()
        base.model_optimizer_stage3.step()
        meter.update({'pid_loss': ide_loss.data,
                      'pid_loss_proj': ide_loss_proj.data,
                      'triplet_loss': triplet_loss.data,
                      'triplet_loss_proj': triplet_loss_proj.data,
                      'rgb_i2t_pid_loss': rgb_i2t_ide_loss.data,
                      'ir_i2t_pid_loss': ir_i2t_ide_loss.data,
                      'msel_loss': msel_loss.data,
                      'msel_loss_proj': msel_loss_proj.data,
                      'inv_loss': inv_loss.data,
                      'center_loss': center_loss.data,
                      'circle_loss': circle_loss.data,
                      })
    return meter.get_val(), meter.get_str()





