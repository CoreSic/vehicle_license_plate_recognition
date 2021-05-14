import cv2
import numpy as np 
import os
from PIL import Image, ImageDraw, ImageFont

img = cv2.imread('D:/My_Resource/class_program/4.python/car_identify/method_2/images/car1.jpg', flags=cv2.IMREAD_GRAYSCALE)
GBlur = cv2.GaussianBlur(img, (3, 3), 0)
canny = cv2.Canny(GBlur, 50, 150)
cv2.imshow('canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()





#-------------------------------------------------------------------------------------------------
#模板匹配
template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z',
            '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁', '蒙', '闽', '宁',
            '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫', '粤', '云', '浙']

# 读取模板文件，读取template文件下的模板
def read_template_file(directory_name):
    template_list = []
    for fileName in os.listdir(directory_name):
        template_list.append(directory_name + "/" + fileName)
    return template_list

# 读取template文件下的模板，开始到结束
def get_template(start, end):
    template_words = []
    for i in range(start, end):
        word = read_template_file('D:/My_Resource/class_program/4.python/car_identify/method_1/template/' + template[i])
        template_words.append(word)
    return template_words

def get_car_No(image, start, end):
    # 模板匹配
    best_scores = []
    template_lists = get_template(start, end)

    for template_list in template_lists:  # 每个文件夹
        scores = []
        for word in template_list:  # 一个文件夹下的多个模板
            template_file = cv2.imdecode(np.fromfile(word, dtype=np.uint8), 1)
            template_img = cv2.cvtColor(template_file, cv2.COLOR_RGB2GRAY)
            ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
            height, width = template_img.shape[:2]

            carNo = cv2.resize(image, (width, height)) #将图片设置成统一大小
            # macthTemplate中图片要与模板尺寸一样大小
            result = cv2.matchTemplate(carNo, template_img, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)  # 获得分值
            scores.append(score)
        best_scores.append(max(scores))

    index = best_scores.index(max(best_scores))  # 分值最大的索引
    return template[start + index]  # 起始序号  索引

car_no_list = []
#-------------------------------------------------------------------------------------------------




#将原图做个备份
sourceImage = img.copy()
#高斯模糊滤波器对图像进行模糊处理
img = cv2.GaussianBlur(img, (3, 3), 0)
#canny边缘检测
img = cv2.Canny(img, 500, 200, 3)
cv2.imshow('Canny', img)
#指定核大小，如果效果不佳，可以试着将核调大
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (29, 1))
kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
#对图像进行膨胀腐蚀处理
img = cv2.dilate(img, kernelX, anchor=(-1, -1), iterations=2)
img = cv2.erode(img, kernelX, anchor=(-1, -1), iterations=4)
img = cv2.dilate(img, kernelX, anchor=(-1, -1), iterations=2)
img = cv2.erode(img, kernelY, anchor=(-1, -1), iterations=1)
img = cv2.dilate(img, kernelY, anchor=(-1, -1), iterations=2)
#再对图像进行模糊处理
img = cv2.medianBlur(img, 15)
img = cv2.medianBlur(img, 15)
cv2.imshow('dilate&erode', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




#检测轮廓，
#输入的三个参数分别为：输入图像、层次类型、轮廓逼近方法
#因为这个函数会修改输入图像，所以上面的步骤使用copy函数将原图像做一份拷贝，再处理
#返回的三个返回值分别为：修改后的图像、图轮廓、层次
contours, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
for c in contours:
    # 边界框
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    print('w' + str(w))
    print('h' + str(h))
    print(float(w)/h)
    print('------')
    #由于国内普通小车车牌的宽高比为3.14，所以，近似的认为，只要宽高比大于2.2且小于4的则认为是车牌
    if float(w)/h >= 2.0 and float(w)/h <= 5.0:
        #将车牌从原图中切割出来
        lpImage = sourceImage[y:y+h, x:x+w]
 
if 'lpImage' not in dir():
    print('未检测到车牌!')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()
 
cv2.imshow('img', lpImage)
cv2.waitKey(0)
cv2.destroyAllWindows()



img_car=lpImage.copy()

#边缘检测
lpImage = cv2.Canny(lpImage, 500, 200, 3)
#对图像进行二值化操作
ret, thresh = cv2.threshold(lpImage.copy(), 127, 255, cv2.THRESH_BINARY)
cv2.imshow('img', thresh)
cv2.waitKey(10)
cv2.destroyAllWindows()


#轮廓检测
contours, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
i = 0
lpchars = []
for c in contours:
    # 边界框
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 0, 0), 2)
 
    print('w' + str(w))
    print('h' + str(h))
    print(float(w)/h)
    print(str(0.8 * thresh.shape[0]))
    print('------')
 
    #根据比例和高判断轮廓是否字符
    if float(w)/h >= 0.3 and float(w)/h <= 0.8 and h >= 0.6 * thresh.shape[0]:
        #将车牌从原图中切割出来
        lpImage2 = lpImage[y:y+h, x:x+w]
        img_car_gray=cv2.threshold(img_car[y:y+h, x:x+w], 127, 255, 0)[1]
        car_no_list.append(img_car_gray)#分割出字符并保存
        cv2.imshow(str(i), img_car_gray)
        i += 1
        lpchars.append([x, y, w, h])
 
cv2.imshow('sdd', thresh)
 
if len(lpchars) < 1:
    print('未检测到字符!')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()
 
lpchars = np.array(lpchars)
#对x坐标升序，这样，字符顺序就是对的了
lpchars = lpchars[lpchars[:,0].argsort()]
print(lpchars)
 
#如果识别的字符小于7，说明汉字没识别出来，要单独识别汉字
if len(lpchars) < 7:
    aveWidth = 0
    aveHeight = 0
    aveY = 0
    for index in lpchars:
        aveY += index[1]
        aveWidth += index[2]
        aveHeight += index[3]
 
    aveY = aveY/len(lpchars)
    aveWidth = aveWidth/len(lpchars)
    aveHeight = aveHeight/len(lpchars)
    zhCharX = lpchars[0][0] - (lpchars[len(lpchars) - 1][0] - lpchars[0][0]) / (len(lpchars) - 1)
    if zhCharX < 0:
        zhCharX = 0
 
    print(aveWidth)
    print(aveHeight)
    print(zhCharX)
    print(aveY)
    #cv2.imshow('img', lpImage[aveY:aveY + aveHeight, zhCharX:zhCharX + aveWidth])
    
    img_car_gray=cv2.threshold(img_car[int(aveY):int(aveY) + int(aveHeight), int(zhCharX):int(zhCharX) + int(aveWidth)+5], 127, 255, 0)[1]
    cv2.imshow('province', img_car_gray)
    car_no_list.append(img_car_gray)#分割出字符并保存
 
cv2.waitKey(0)
cv2.destroyAllWindows()

#-------------------------------------------------------------------------------------------------
car_no_list.reverse()
car_all_no = []
# 第一个汉字
first_chinese = get_car_No(car_no_list[0], 34, 64)
car_all_no.append(first_chinese)
# 第二个英文字母
second_english = get_car_No(car_no_list[1], 10, 33)
car_all_no.append(second_english + " ")
# 数字及英文字母
for car_no in car_no_list[2:]:
    number_english = get_car_No(car_no, 0, 33)
    car_all_no.append(number_english)
print(car_all_no)

# 显示中文
def cv2ImgAddText(img, text, left, top, textColor=(255, 255, 0), textSize=50):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("font/STSONG.TTF", textSize, encoding="utf-8")  # 字体
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

result_img = cv2ImgAddText(sourceImage, "".join(car_all_no), 10, 20)
cv2.imshow('result_img', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#-------------------------------------------------------------------------------------------------
