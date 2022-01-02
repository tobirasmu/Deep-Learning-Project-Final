def intersection_over_union(boxA,boxB):
    # calculates Intersection over Union for two boxes
    # INPUT
    # boxA/boxB : bounding boxes [minx,maxx,miny,maxy]
    # OUTPUT
    # iou : Intersection over Union of the input bounding boxes.
    
    # calculate intersection box
    minx = max(boxA[0],boxB[0])
    maxx = min(boxA[1],boxB[1])
    miny = max(boxA[2],boxB[2])
    maxy = min(boxA[3],boxB[3])
    
    # check if overlap == 0
    if minx > maxx or miny > maxy:
        return 0
    
    # intersection area
    intersection = (maxx-minx)*(maxy-miny)
    
    # union area
    areaA = (boxA[1]-boxA[0])*(boxA[3]-boxA[2])
    areaB = (boxB[1]-boxB[0])*(boxB[3]-boxB[2])
    union = areaA+areaB-intersection
    
    # calculate intersection over union
    iou = intersection/union
    return iou
   
if __name__ == "__main__":
    print("testing IoU function")
    bA = [1,2,1,2]
    bB = [1.5,2,1.5,2]
    iou = intersection_over_union(bA, bB)
    print('boxA:',bA)
    print('boxB:',bB)
    print('IoU:',iou,' Expected: 0.25\n')
    bA = [1,2,1,2]
    bB = [2.5,2,1.5,2]
    iou = intersection_over_union(bA, bB)
    print('boxA:',bA)
    print('boxB:',bB)
    print('IoU:',iou,' Expected: 0\n')
    bA = [1,2,1,2]
    bB = [1,2,1,2]
    iou = intersection_over_union(bA, bB)
    print('boxA:',bA)
    print('boxB:',bB)
    print('IoU:',iou,' Expected: 1\n')
    bA = [1,2,1,2]
    bB = [0,3,0,3]
    iou = intersection_over_union(bA, bB)
    print('boxA:',bA)
    print('boxB:',bB)
    print('IoU:',iou,' Expected: 0.11..')
    
