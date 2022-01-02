import json


def add_bounding_box(json_file,output_file):
    """ Add a bounding_boxes element to json vertice data imported from Zendo
	results are written to a new json file """
    with open(json_file) as f:
        data = json.load(f)
    for obj in data['objects']:
        #label = obj['labels']['label']
        vert = obj['mask_vertices'][0]
        y = [v[0] for v in vert]
        x = [v[1] for v in vert]
        ymin = min(y)
        ymax = max(y)
        xmin = min(x)
        xmax = max(x)
        obj['bounding_boxes'] = [[xmin,xmax,ymin,ymax]]
    with open(output_file,'w') as f:
        json.dump(data,f,indent=4)
    return data
        

def test():
    filename = "vlcsnap-2021-10-10-21h35m49s561.json"
    imgname = "vlcsnap-2021-10-10-21h35m49s561.jpeg"
    data = add_bounding_box(filename,filename[:-5]+"-new.json")
    
    img = matplotlib.image.imread(imgname)
    figure, ax = matplotlib.pyplot.subplots(1)
    ax.imshow(img)
    
    for obj in data['objects']:
        bb = obj['bounding_boxes'][0]
        rect = matplotlib.patches.Rectangle((bb[2],bb[0]),bb[3]-bb[2],bb[1]-bb[0], edgecolor='r', facecolor="none")
        ax.add_patch(rect)
    return data


if __name__ == "__main__":
    import matplotlib
    test()
    
