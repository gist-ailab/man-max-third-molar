import json



with open("./dataset/Annotations/4025160000.jpg.png.json") as json_file:
    json_data = json.load(json_file)
    json_obj = json_data["objects"]
    img_label = []

    for anno_num in range(len(json_obj)):
        object_class_id = json_data["objects"][anno_num]["classId"]
        object_class_Title = json_data["objects"][anno_num]["classTitle"]
        object_class_tags = json_data["objects"][anno_num]["tags"]
        if object_class_tags == 0:
            continue
        
        else:
            for tag in range(len(object_class_tags)):

                if object_class_Title == "#48":
                    object_class_name = json_data["objects"][anno_num]["tags"][tag]["name"]
                    img_label.append(object_class_name)
                else:
                    pass
        print(object_class_id)
        print(object_class_Title)
        print(img_label)
