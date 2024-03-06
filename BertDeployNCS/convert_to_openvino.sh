#!/bin/sh
PATH_TO_OPENVINO=/opt/intel/openvino_2021
echo Converting to IR model
python3 $PATH_TO_OPENVINO/deployment_tools/model_optimizer/mo.py --input_model $1 --input=input_ids,attention_mask,token_type_ids --input_shape=[1,128],[1,128],[1,128]
mv *.xml movidius_models/
mv *.bin movidius_models/
mv *.mapping movidius_models/
