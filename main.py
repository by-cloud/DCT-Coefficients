import numpy as np
import matplotlib.pyplot as plt
import cv2


# 读取图片
image = cv2.imread('test.jpeg', 0)
# 离散余弦变换，并获取其幅频谱
img_dct = cv2.dct(np.float32(image))
img_dct_log = np.log(abs(img_dct))
# 逆离散余弦变换，变换图像至空间域
img_back = cv2.idct(img_dct)
print(type(img_dct))

ret = image * img_dct
result = np.zeros((8, 8))

for i in range(0, 256):
    for j in range(0, 256):
        result[i // 32][j // 32] += ret[i][j]

result = result / (32 * 32)
np.set_printoptions(suppress=True)
print(result)

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('original Lena', fontproperties='Times New Roman')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_dct_log, cmap='gray')
plt.title('DCT of Lena', fontproperties='Times New Roman')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_back, cmap='gray')
plt.title('iDCT of Lena', fontproperties='Times New Roman')
plt.axis('off')

plt.savefig('lena_dct.png', dpi=300)
plt.show()

plt.close()
