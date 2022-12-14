class DefaultConfig():
    #backbone
    pretrained=True
    freeze_stage_1=False  #True
    freeze_bn=False

    #fpn
    fpn_out_channels=256
    use_p5=True
    
    #head
    class_num=1
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True

    #training
    strides=[4,8,16,32,64,128]
    limit_range=[[0,8],[8,16],[16,32],[32,64],[64,128],[128,9999999]]

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.5
    max_detection_boxes_num=1