import os
from PIL import Image
import cv2
import numpy as np

def overlay_davis(image, mask,colors=[255,0,0],cscale=2,alpha=0.4):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # import skimage
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale
    colors = [255, 255, 0]
    # print(colors)
    im_overlay = image.copy()
    object_ids = np.unique(mask)
    # print(object_ids)

    for object_id in object_ids[1:]:
        # print(object_id)
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors)
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours,:] = 0

    return im_overlay.astype(image.dtype)


def save_mask(image, location, mask, mask_real, segm_crop_sz, bb, img_w, img_h, masks_save_path, sequence_name, frame_name):
    if mask is not None:
        M_sel = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
        mask_resized = (cv2.resize((M_sel * mask_real).astype(np.float32), (segm_crop_sz, segm_crop_sz),
                                   interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    else:
        mask_resized = (cv2.resize(mask_real.astype(np.float32), (segm_crop_sz, segm_crop_sz),
                                   interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    image_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    # patch coordinates
    xp0 = 0
    yp0 = 0
    xp1 = mask_resized.shape[0]
    yp1 = mask_resized.shape[1]
    # image coordinates
    xi0 = int(round((bb[0] + bb[2] / 2) - mask_resized.shape[1] / 2))
    yi0 = int(round((bb[1] + bb[3] / 2) - mask_resized.shape[0] / 2))
    xi1 = int(round(xi0 + mask_resized.shape[1]))
    yi1 = int(round(yi0 + mask_resized.shape[0]))
    if xi0 < 0:
        xp0 = -1 * xi0
        xi0 = 0
    if xi0 > img_w:
        xp0 = (mask_resized.shape[1]) - (xi0 - img_w)
        xi0 = img_w
    if yi0 < 0:
        yp0 = -1 * yi0
        yi0 = 0
    if yi0 > img_h:
        yp0 = (mask_resized.shape[0]) - (yi0 - img_h)
        yi0 = img_h
    if xi1 < 0:
        xp1 = -1 * xi1
        xi1 = 0
    if xi1 > img_w:
        xp1 = (mask_resized.shape[1]) - (xi1 - img_w)
        xi1 = img_w
    if yi1 < 0:
        yp1 = -1 * yi1
        yi1 = 0
    if yi1 > img_h:
        yp1 = (mask_resized.shape[0]) - (yi1 - img_h)
        yi1 = img_h

    image_mask[yi0:yi1, xi0:xi1] = mask_resized[yp0:yp1, xp0:xp1]

    # COLORS = [255, 255, 0]
    # # COLORS = np.random.randint(128, 255, size=(1, 3), dtype="uint8")
    # # print(COLORS)
    # COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
    # image_mask1 = COLORS[image_mask]
    # image_mask1 = cv2.cvtColor(image_mask1, cv2.COLOR_BGR2RGB)


    mask_save_dir = os.path.join(masks_save_path, sequence_name)
    if not os.path.exists(mask_save_dir):
        os.mkdir(mask_save_dir)

    palette = Image.open(os.path.join('/home/yuhongtao/d3sv1/assets/mask_palette.png')).getpalette()
    # print(palette)
    pF = (image).astype(np.uint8)
    pE = image_mask
    canvas = overlay_davis(pF, pE, palette)
    canvas = Image.fromarray(canvas)

##################################################
    canvas = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    if len(location) == 8:
        location_int = np.int0(location)
        cv2.polylines(canvas, [location_int.reshape((-1, 1, 2))], True, (0, 255, 0), 3)
    # else:
    #     location = [int(l) for l in location]
    #     cv2.rectangle(canvas, (location[0], location[1]),
    #                   (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 2)
    # cv2.putText(canvas, str(frame_name), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # canvas.save(os.path.join(mask_save_dir, '{}.png'.format(frame_name)))

    mask_save_path = os.path.join(mask_save_dir, '%s.png' % frame_name)
    cv2.imwrite(mask_save_path, canvas)


    # mask_save_dir = os.path.join(masks_save_path, sequence_name)
    # if not os.path.exists(mask_save_dir):
    #     os.mkdir(mask_save_dir)
    # mask_save_path = os.path.join(mask_save_dir, '%s.png' % frame_name)
    # cv2.imwrite(mask_save_path, image_mask1)
    return image_mask