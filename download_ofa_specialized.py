# Download Specialized OFA

hw2ofa={
    "XEON_BS1":
    [
        'cpu_lat@17ms_top1@75.7_finetune@25',
        'cpu_lat@15ms_top1@74.6_finetune@25',
        'cpu_lat@11ms_top1@72.0_finetune@25',
        'cpu_lat@10ms_top1@71.1_finetune@25',
    ],
    "V100_BS64": 
    [
        'v100_gpu64@11ms_top1@76.1_finetune@25',
        'v100_gpu64@9ms_top1@75.3_finetune@25',
        'v100_gpu64@6ms_top1@73.0_finetune@25',
        'v100_gpu64@5ms_top1@71.6_finetune@25'
    ],
    "GTX1080TI":
    [
        '1080ti_gpu64@27ms_top1@76.4_finetune@25',
        '1080ti_gpu64@22ms_top1@75.3_finetune@25',
        '1080ti_gpu64@15ms_top1@73.8_finetune@25',
        '1080ti_gpu64@12ms_top1@72.6_finetune@25',  
    ]    
}

from model_zoo import ofa_specialized

for hw, ofa_list in hw2ofa.items():
    for net_id in ofa_list:
        net, image_size = ofa_specialized(net_id, pretrained=True)
        print("Downloaded | {} | with required input dim {}\n".format(net_id, image_size))