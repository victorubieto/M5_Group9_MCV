import json
import numpy as np
import matplotlib.pyplot as plt

with open("instances_train2017.json", "r") as read_file:
   data = json.load(read_file)

imgsdata = {}
instances_count = {}
for i in range(len(data['annotations'])):
    img_id = data['annotations'][i]['image_id']
    category_id = data['annotations'][i]['category_id']
    if img_id in imgsdata:
        imgsdata[img_id].append(category_id)
    else:
        imgsdata[img_id] = [category_id]
    if category_id in instances_count:
        instances_count[category_id] += 1
    else:
        instances_count[category_id] = 1

sorted_instances_count = dict(sorted(instances_count.items()))

unique_imgsdata = {}
keys = [*imgsdata.keys()]
for key in keys:
    unique_imgsdata[key] = np.unique(imgsdata[key])

conf_matrix = np.zeros((91,91))
keys = [*unique_imgsdata.keys()]
for key in keys:
    categories = unique_imgsdata[key]
    for i in range(len(categories)-1):
        cat = categories[i]
        for paired_cat in categories[i+1:]:
            conf_matrix[cat][paired_cat] += 1
            conf_matrix[paired_cat][cat] += 1


print("Number of images: " + str(len(imgsdata.keys())))
print(sorted_instances_count)
print(conf_matrix)
plt.matshow(conf_matrix)
plt.show()
a = 0
