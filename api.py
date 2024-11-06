import os

import cv2
import numpy as np

from utils.aux_functions import *


def from_300w_format(points):
    landmarks = {
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
    }
    return landmarks


def mask(image, bbox, landmarks, mask_type=None, pattern=None,
         pattern_weight=None, color=None, color_weight=None, debug=True):
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    x1, y1, x2, y2 = bbox
    if mask_type is None:
        mask_type = random.choice(get_available_mask_types())
    six_points, angle = get_six_points(landmarks, image)
    threshold = 13
    if angle < -threshold:
        mask_type += "_right"
    elif angle > threshold:
        mask_type += "_left"
    face_height = y2 - y1
    face_width = x2 - x1
    h, w = image.shape[:2]
    if not "empty" in mask_type and not "inpaint" in mask_type:
        cfg = read_cfg(
            config_filename=os.path.join(current_dir, "masks/masks.cfg"),
            mask_type=mask_type, verbose=False)
    else:
        if "left" in mask_type:
            str = "surgical_blue_left"
        elif "right" in mask_type:
            str = "surgical_blue_right"
        else:
            str = "surgical_blue"
        cfg = read_cfg(
            config_filename=os.path.join(current_dir, "masks/masks.cfg"),
            mask_type=str, verbose=False)
    img = cv2.imread(os.path.join(current_dir, cfg.template), cv2.IMREAD_UNCHANGED)

    # Process the mask if necessary
    if pattern:
        # Apply pattern to mask
        img = texture_the_mask(img, pattern, pattern_weight)

    if color:
        # Apply color to mask
        img = color_the_mask(img, color, color_weight)

    mask_line = np.float32(
        [cfg.mask_a, cfg.mask_b, cfg.mask_c, cfg.mask_f, cfg.mask_e, cfg.mask_d]
    )
    # Warp the mask
    M, mask = cv2.findHomography(mask_line, six_points)
    dst_mask = cv2.warpPerspective(img, M, (w, h))
    dst_mask_points = cv2.perspectiveTransform(mask_line.reshape(-1, 1, 2), M)
    mask = dst_mask[:, :, 3]
    image_face = image[y1 + int(face_height / 2):y2, x1:x2, :]

    image_face = image

    # Adjust Brightness
    mask_brightness = get_avg_brightness(img)
    img_brightness = get_avg_brightness(image_face)
    delta_b = 1 + (img_brightness - mask_brightness) / 255
    dst_mask = change_brightness(dst_mask, delta_b)

    # Adjust Saturation
    mask_saturation = get_avg_saturation(img)
    img_saturation = get_avg_saturation(image_face)
    delta_s = 1 - (img_saturation - mask_saturation) / 255
    dst_mask = change_saturation(dst_mask, delta_s)

    # Apply mask
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(image, image, mask=mask_inv)
    img_fg = cv2.bitwise_and(dst_mask, dst_mask, mask=mask)
    out_img = cv2.add(img_bg, img_fg[:, :, 0:3])
    if "empty" in mask_type or "inpaint" in mask_type:
        out_img = img_bg
    # Plot key points

    if "inpaint" in mask_type:
        out_img = cv2.inpaint(out_img, mask, 3, cv2.INPAINT_TELEA)
        # dst_NS = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    if debug:
        six_points = six_points.astype(np.int32)
        dst_mask_points = dst_mask_points.astype(np.int32)
        for i in six_points:
            cv2.circle(out_img, (i[0], i[1]), radius=4, color=(0, 0, 255), thickness=-1)

        for i in dst_mask_points:
            cv2.circle(
                out_img, (i[0][0], i[0][1]), radius=4, color=(0, 255, 0), thickness=-1
            )
    return out_img, mask


if __name__ == "__main__":
    image = cv2.imread("./images/afw_1051618982_1.jpg")
    kps = []
    with open("./images/afw_1051618982_1.pts") as file:
        lines = file.read().splitlines()[3:-1]
        for line in lines:
            x, y = line.split(" ")
            kps.append([round(float(x)), round(float(y))])
    kps_arr = np.array(kps)
    x1 = kps_arr[:, 0].min()
    x2 = kps_arr[:, 0].max()
    y1 = kps_arr[:, 1].min()
    y2 = kps_arr[:, 1].max()
    landmarks = from_300w_format(kps)
    out_img, mask = mask(image, (x1, y1, x2, y2), landmarks,
         mask_type=None,
         pattern=None, pattern_weight=None,
         color=None, color_weight=None,
         debug=True)
    cv2.imshow("out", out_img);
    cv2.waitKey()
    cv2.imshow("mask", mask);
    cv2.waitKey()

