# Category_Inference_FL

```
@article{gao2021secure,
  title={Secure Aggregation is Insecure: Category Inference Attack on Federated Learning},
  author={Gao, Jiqiang and Hou, Boyu and Guo, Xiaojie and Liu, Zheli and Zhang, Ying and Chen, Kai and Li, Jin},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2021},
  publisher={IEEE}
}
```



This project provides FL category inference attack code on MNIST datasets, including noise-logger strategy.  If you need additional code in the paper, you can contact us by emailing or making an issue. 

## Requirements

* Python >=3.6
* Pytorch >=1.1.0
* torchvision >=0.3.0
* cuda >=10.0
* cudnn >=7.6.5

## Running

```shell
cd Category_Inference_FL
# inference attack with noise-logger on MNIST 
python clean_attack_release.py
```

