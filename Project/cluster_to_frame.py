import cv2

c=0
img=0
while c<5:
    file_name = "./clusters/frame0/cluster"+str(c)+".png"

    src = cv2.imread(file_name, 1)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)

    img+=dst

    c+=1


cv2.imwrite("./frames_WC/test.png", img)
