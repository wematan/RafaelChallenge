import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def show(image):
    cv2.imshow('temp', image)
    cv2.waitKey(0)


def plot(image, color_map='gray'):
    plt.subplot(111), plt.imshow(image, cmap=color_map)
    plt.show()


class Question:
    def __init__(self, img, type='png'):
        self.img = img
        if type == 'png':
            self.img_obj = cv2.imread(self.img, -1)
        elif type == 'gif' or type == 'PIL_png':
            self.img_obj = Image.open(self.img)
        elif type == 'vid':
            self.img_obj = cv2.VideoCapture(self.img)


# Question 1
class Q1(Question):
    def solve(self):
        bright_img = self.selective_brightness(self.img_obj)
        show(bright_img)
        return

    def selective_brightness(self, img, value=100):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v==1] += lim

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img


# Question 2
class Q2(Question):
    def filters(self, img):
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        return [laplacian]

    def solve(self):
        # Remove hidden alpha channel (You Bastards!)
        img = self.img_obj
        alpha_channel = img[:, :, 3]
        _, mask = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)  # binarize mask
        color = img[:, :, :3]
        new_img = cv2.bitwise_not(cv2.bitwise_not(color, mask=mask))
        ret = self.filters(new_img)
        for i in ret:
            plot(i)
        return

# Question 3
class Q3(Question):
    def play_with_palette(self):
        palette = self.img_obj.getpalette()
        # Palette index 164 (each palette index has 3 values [r,g,b])
        palette[492] = palette[493] = palette[494] = 255
        img = self.img_obj
        img.putpalette(palette)
        plot(img)
        return

    def solve(self):
        print "The Clue is in the gif info..."
        print self.img_obj.info['comment']
        self.play_with_palette()
        return

# Question 4
class Q4(Question):

    def img_iter(self):
        f, i = self.img_obj.read()
        while f:
            yield i
            f, i = self.img_obj.read()

    def solve(self):
        index = 0
        img = self.img_iter().next()
        for frame in self.img_iter():
            img = cv2.addWeighted(img, 0.5, frame, 0.5, 1)
        show(img)
        return

# Question 5
class Q5(Question):
    def morph(self, img, kernel_size):
        img = np.asarray(img)
        ret, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
        return morph_img

    def solve(self):
        r_img1 = self.img_obj.resize((1000, 800))
        show(self.morph(r_img1, (1,2)))
        r_img2 = self.img_obj.resize((1500, 900))
        show(self.morph(r_img2, (1,2)))

        return

class Q6(Question):
    def __init__(self, img1, img2, type='gif'):
        self.img1 = img1
        self.img2 = img2
        if type == 'gif':
            self.img_obj1 = Image.open(self.img1)
            self.img_obj2 = Image.open(self.img2)
        elif type == 'vid':
            self.img_obj = cv2.VideoCapture(self.img)

    def fft(self, img):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 2 * np.log(np.abs(fshift))
        return np.uint8(magnitude_spectrum)

    def solve(self):
        img1 = np.asarray(self.img_obj1).astype('float')
        img2 = np.asarray(self.img_obj2).astype('float')
        cplx_img = np.zeros(img1.shape, dtype='complex')
        # cplx_img = np.sqrt(img1**2, img2**2)
        for i in range(0, self.img_obj1.size[0]):
            for j in range(0, self.img_obj1.size[1]):
                cplx_img[i, j] = self.img_obj1.getpixel((i, j)) + self.img_obj2.getpixel((i, j))
        # # real
        # self.img_obj1
        # # imaginary
        # self.img_obj2

        img = self.fft(cplx_img)
        img = np.flipud(np.rot90(img, axes=(-2, -1)))
        a = img[100:,:]
        img = np.concatenate((a , img), axis=0)
        a = img[:, 100:]
        img = np.concatenate((a, img), axis=1)

        plot(img)

        return


class Q7(Question):
    def __init__(self, img1, img2, type='gif'):
        self.img1 = img1
        self.img2 = img2
        if type == 'gif':
            self.img_obj1 = Image.open(self.img1)
            self.img_obj2 = Image.open(self.img2)
        elif type == 'vid':
            self.img_obj = cv2.VideoCapture(self.img)

    def solve(self):
        img1 = np.asarray(self.img_obj1)
        gray = np.asarray(self.img_obj1.convert('L'))
        template = np.rot90(np.asarray(self.img_obj2.convert('L')))
        w, h = template.shape[::-1]

        response = cv2.matchTemplate(gray, template, cv2.TM_CCORR_NORMED)

        _, _, _, maxLoc = cv2.minMaxLoc(response)

        threshold = 0.90
        loc = np.where(response >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img1, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

        print "Q7 Answer %d#%d" % (maxLoc)

        return

class Q8(Question):
    def solve(self):
        img = np.array(self.img_obj)
        red_pixels_mask = np.all(img == [255, 0, 0], axis=-1)
        lineDict = {}
        for line in range(red_pixels_mask.shape[0]):
            lineDict[np.argwhere(red_pixels_mask[line, :])[0][0]] = line
        res = np.zeros(img.shape).astype(np.uint8)
        sortedKeys = sorted(lineDict.keys(), reverse=True)
        for y in range(len(sortedKeys)):
            res[y, :, :] = img[lineDict[sortedKeys[y]], :, :]
        show(res)
        return

if __name__ == '__main__':
    os.chdir('Questions')

    q1 = Q1("ch1.png")
    q1.solve()

    q2 = Q2("ch2.png", 'png')
    q2.solve()

    q3 = Q3("ch3.gif", type='gif')
    q3.solve()

    q4 = Q4("ch4.gif", type='vid')
    q4.solve()

    q5 = Q5("ch5.gif", type='gif')
    q5.solve()

    q6 = Q6("ch6_1.gif", "ch6_2.gif", type='gif')
    q6.solve()

    q7 = Q7("ch7_1.gif", "ch7_2.gif", type='gif')
    q7.solve()

    q8 = Q8("ch8.png", type='PIL_png')
    q8.solve()