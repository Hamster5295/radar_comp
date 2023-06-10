import numpy as np


# def filter(img):
#     dtype = 'float32'
#     w, h = img.shape
#     window_size = 7
#     half_wf = window_size // 2
#     m = np.zeros((3, 3), dtype=dtype)
#     img_filtered_rl = np.zeros((w - half_wf, h - half_wf))
#     g1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype)
#     g2 = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], dtype)
#     g3 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype)
#     g4 = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], dtype)
#
#     for i in range(half_wf, w - half_wf):
#         for j in range(half_wf, h - half_wf):
#             # print(i)
#             window = img[i - half_wf:i + half_wf + 1, j - half_wf:j + half_wf + 1]
#             for k in range(3):
#                 for l in range(3):
#                     m[k, l] = np.mean(window[k * 2:k * 2 + 2, l * 2:l * 2 + 2])
#             gv = np.zeros((1, 4), dtype)
#             gv[0, 0] = np.sum(np.multiply(g1, m))
#             gv[0, 1] = np.sum(np.multiply(g2, m))
#             gv[0, 2] = np.sum(np.multiply(g3, m))
#             gv[0, 3] = np.sum(np.multiply(g4, m))
#             gc = np.argmax(gv)
#             gv = np.max(gv)
#             part = []
#             if gc == 0:
#                 part = window[0:7, 3:7]
#                 if np.abs(m[1, 0] - m[1, 1]) < np.abs(m[1, 2] - m[1, 1]):
#                     part = window[0:7, 0:3]
#             elif gc == 1:
#                 part = np.vstack(
#                     (window[0:1, 0:1], window[1:2, 1:2], window[2:3, 2:3], window[3:4, 3:4], window[4:5, 4:5],
#                      window[5:6, 5:6], window[6:7, 6:7]))
#                 if np.abs(m[2, 0] - m[2, 2]) < np.abs(m[0, 2] - m[1, 1]):
#                     part = np.vstack(
#                         (window[0:7, 0:1], window[1:7, 1:2], window[2:7, 2:3], window[3:7, 3:4], window[4:7, 4:5],
#                          window[5:7, 5:6], window[6:7, 6:7]))
#             elif gc == 2:
#                 part = window[0:4, 0:7]
#                 if np.abs(m[0, 1] - m[1, 1]) < np.abs(m[2, 1] - m[1, 1]):
#                     part = window[3:7, 0:7]
#             else:
#                 # print(window[6:7, 0:1].shape, window[5:7, 1:2].shape)
#                 part = np.vstack(
#                     (window[6:7, 0:1], window[5:7, 1:2], window[4:7, 2:3], window[3:7, 3:4], window[2:7, 4:5],
#                      window[1:7, 5:6], window[0:7, 6:7]))
#                 if np.abs(m[0, 0] - m[1, 1]) < np.abs(m[2, 2] - m[1, 1]):
#                     part = np.vstack(
#                         (window[0:7, 0:1], window[0:7, 1:2], window[0:5, 2:3], window[0:4, 3:4], window[0:3, 4:5],
#                          window[0:2, 5:6], window[0:1, 6:7]))
#
#                 # print(part.shape)
#                 pw, ph = part.shape[0], part.shape[1]
#                 temp = np.reshape(part, (1, pw * ph))
#                 u_y = np.mean(part)
#                 var_y = np.std(temp)
#                 var_x = (var_y - u_y * u_y) / 2
#                 a = 1 - var_x / var_y
#                 b = var_x / var_y
#                 img_filtered_rl[i - half_wf + 1, j - half_wf + 1] = u_y + b * (img[i, j] - u_y)
#     img_filtered_rl /= np.max(img_filtered_rl)
#
#     img_filtered = imadjust(img_filtered_rl)
#     return img_filtered


def apply_window(img):
    w, h = img.shape

    wc = np.hamming(h)
    wr = np.hamming(w)
    maskr, maskc = np.meshgrid(wc, wr)
    window = maskr * maskc
    return img * window
