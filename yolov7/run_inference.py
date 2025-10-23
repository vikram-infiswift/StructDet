"""
Optimized YOLOv7 inference for component/table detection and cropping.
Focus: memory + speed optimized, identical functionality.
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision
import time


# Helpers
def scale_coords(new_shape, coords, orig_shape):
    """Scales detection coords from resized -> original image size."""
    orig_w, orig_h, _ = orig_shape
    new_w, new_h = new_shape

    # Compute scaling factors
    gain_w = orig_w / new_w
    gain_h = orig_h / new_h

    coords[:, [0, 2]] *= gain_h  # scale x1, x2
    coords[:, [1, 3]] *= gain_w  # scale y1, y2
    return coords


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def auto_pad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, auto_pad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        # attempt_download(w)
        ckpt = torch.load(w, map_location=map_location, weights_only=False)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


def attempt_load_jit(weights, map_location=None):
    model = torch.jit.load(str(weights), map_location=map_location)
    return model


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5]  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
            # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


@torch.no_grad()
def run_inference(
        folder_dir: Path,
        model_path: Path,
        output_dir: Path,
        batch_size=4,
        img_size=(640, 640),
        conf_thresh=0.3,
        iou_thresh=0.65,
):
    # ------------------- Device -------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    half = device.type != "cpu"

    # ------------------- Load model -------------------
    model = attempt_load_jit(model_path, map_location=device)
    model.eval()
    if half:
        model.half()

    # names = model.module.names if hasattr(model, "module") else model.names
    names = getattr(model, "names", ["component", "table"])  # fallback if not saved
    print(f"Classes: {names}")
    print(f"✅ Model loaded with classes: {names}")

    # ------------------- Prepare folder -------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------- Load images -------------------
    image_files = [p for p in Path(folder_dir).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not image_files:
        print("❌ No images found in", folder_dir)
        return

    imgs, im0_shapes, paths = [], [], []

    # ------------------- Process in batches -------------------
    for img_path in image_files:
        # --- Preprocess ---
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Resize using LANCZOS
        image_resized = image.resize(img_size, Image.Resampling.LANCZOS)
        img = np.array(image_resized, dtype=np.uint8)[:, :, ::-1].transpose(2, 0, 1).copy()

        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.half() if half else img_tensor.float()
        img_tensor /= 255.0

        imgs.append(img_tensor)
        im0_shapes.append((orig_h, orig_w))
        paths.append(str(img_path))

        if len(imgs) == batch_size:
            process_batch(model, imgs, im0_shapes, paths, output_dir, names, conf_thresh, iou_thresh, img_size)
            imgs.clear();
            im0_shapes.clear();
            paths.clear()

    # ------------------- Process remaining -------------------
    if imgs:
        process_batch(model, imgs, im0_shapes, paths, output_dir, names, conf_thresh, iou_thresh, img_size)


def process_batch(model, imgs, im0_shapes, paths, cropped_root, names, conf_thresh, iou_thresh, img_size):
    """Run inference on a batch, save cropped components and annotated images inside per-image folders."""
    batch = torch.stack(imgs)
    preds = model(batch)[0]
    preds = non_max_suppression(preds, conf_thresh, iou_thresh)

    for det, im_shape, p in zip(preds, im0_shapes, paths):
        im0 = cv2.imread(p)
        # Copy for visualization (BB drawn here)
        im_vis = im0.copy()
        if im0 is None:
            continue

        image_name = Path(p).stem
        image_save_dir = cropped_root / image_name
        image_save_dir.mkdir(parents=True, exist_ok=True)

        detect_save_path = image_save_dir / f"{image_name}_yolo_detection.png"

        if len(det):
            det[:, :4] = scale_coords(img_size, det[:, :4], im0.shape).round()
            for i, (*xyxy, conf, cls) in enumerate(reversed(det)):
                x1, y1, x2, y2 = map(int, xyxy)
                cls_name = names[int(cls)]
                label = f"{cls_name} {conf:.2f}"

                # Draw bounding box and label
                cv2.rectangle(im_vis, (x1, y1), (x2, y2), (255, 0, 0), 10)
                cv2.putText(im_vis, label, (x1, max(y1 - 25, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 5)

                # Crop and save
                crop = im0[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_path = image_save_dir / f"expanded_{cls_name.lower()}_{i}.png"
                cv2.imwrite(str(crop_path), crop)

        # Save image with all bounding boxes
        cv2.imwrite(str(detect_save_path), im_vis)


if __name__ == "__main__":
    # Example usage
    folder_dir = Path("../database/vs_outputs/Residential - 8 level/Paynters-402503/Struct/images")
    model_path = Path("runs/train/struct_det/weights/best.torchscript.pt")
    output_dir = Path("../database/vs_outputs/Residential - 8 level/Paynters-402503/Struct/cropped_components_yolo")

    run_inference(
        folder_dir,
        model_path,
        output_dir,
        batch_size=4,
        img_size=(640, 640),
        conf_thresh=0.3,
        iou_thresh=0.65,
    )
