# ES-HHA
A novel evolutionary status guided hyper-heuristic algorithm for continuous optimization

## Abstract
This paper proposes a novel evolutionary status guided hyper-heuristic algorithm named ES-HHA for continuous optimization. A representative hyper-heuristic algorithm consists of two components: the low-level component and the high-level component. In the low-level component, to balance the exploitation and exploration during optimization, we design an exploitative operator pool and an explorative operator pool as low-level heuristics (LLHs), where the former is constructed using local search based operators, and the latter consists of various mutation operators from differential evolution (DE). In the high-level component, we design a probabilistic selection function based on the fitness distance correlation (FDC) and the population diversity (PD). Since these two metrics can reflect the complexity of the fitness landscape and the status of the evolutionary swarm, the integration of these two metrics is expected to determine the sequence of heuristics automatically and intelligently. To evaluate the performance of our proposal, we implement comprehensive numerical experiments on CEC2014, CEC2022, and eight engineering optimization tasks. A total of 14 famous optimization approaches are adopted as competitors. Furthermore, the ablation experiment is conducted to evaluate the high-level component independently, while the sensitivity experiment contributes to determining the optimal hyperparameter setting. The experimental results and statistical analysis show that ES-HHA is competitive, and the evolutionary status guided probabilistic selection function can determine the optimization intelligently.

## Citation
@article{Zhong:24,  
  title={A novel evolutionary status guided hyper-heuristic algorithm for continuous optimization},  
  author={Zhong, Rui and Yu, Jun},  
  journal={Cluster Computing},  
  volume={27},  
  pages={12209–12238},  
  year={2024},  
  publisher={Springer},  
  doi = {https://doi.org/10.1007/s10586-024-04593-2 },  
}

## Datasets and Libraries
CEC benchmarks and Engineering problems are provided by opfunu==1.0.0 and enoppy==0.1.1 libraries, respectively. 

## Contact
If you have any questions, please don't hesitate to contact zhongrui[at]iic.hokudai.ac.jp
