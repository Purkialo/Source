with open("/home/student/pydata/train/VOC_COCO/ImageSets/Main/trainval.txt","r") as f1:
    with open("/home/student/pydata/train/VOC_COCO/ImageSets/Main/person_trainval.txt","w") as f2:
        for line in f1:
            line=f1.readline()
            f2.write(line[:-1]+'  1'+"\n")