from PIL import Image

if __name__ == "__main__":
    img = Image.open("./input/facial-keypoints-detection/photos/image-13.jpg").convert('L')
    data = list(img.getdata())
    print(len(data))
    textfile = open("./input/facial-keypoints-detection/image_pixel_values/image_pixel_values.txt", "w")
    for element in data:
        textfile.write(str(element) + ' ')
    textfile.close()
