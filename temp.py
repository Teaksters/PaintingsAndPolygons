from PIL import Image
import numpy as np


def img_to_array():
    '''Casts image to a numpy array.'''
    img = Image.open('paintings/monalisa-240-180.png')
    arr = np.array(img)
    return arr


def save_img(img_arr):
    '''Store an image of the current state'''
    img_name = "inv_mona.png"
    im = Image.fromarray(np.uint8(img_arr))
    im.save(img_name)


def main():
    arr = img_to_array()
    inv = []
    for row in arr:
        for pix in row:
            n_pix = [0, 0, 0]
            for i in range(len(pix[:3])):
                if pix[i] <= 127:
                    n_pix[i] = 255
            inv.append(n_pix)
    inv = np.array(inv)
    inv = np.reshape(inv, (240, 180, 3))
    save_img(inv)


if __name__=='__main__':
    main()
