from PIL import Image

def imageblend(imagelist, size=imagelist[0].size):
    defaultimage = Image.new("RGBA", size)
    for i in imagelist:
        defaultimage = Image.alpha_composite(defaultimage, i)
    return defaultimage

def readImage(pathlist):
    imagelist = []
    for i in pathlist:
        imagelist.append(Image.open(str(i)).convert('RGBA'))
    return imagelist

def generate(pathlist):
    return imageblend(readImage(pathlist))