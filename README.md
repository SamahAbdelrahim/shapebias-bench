# shapebias-bench
A benchmark for evaluating the shape bias in CNN and VLM 

two folders:
stimuli_pipe to create stimuli 
evaluation and model pipeline 


```
shapebias-bench/                                                                                                                                      
  ├── README.md                                                                                                                                                                      
  ├── .gitignore                                                                                                                                      
  ├── configs/
  │   ├── experiment_default.yaml
  │   └── prompts.yaml
  ├── evaluation_pipe/
  │   ├── models/
  │   │   ├── __init__.py
  │   │   ├── base.py
  │   │   ├── local_models/
  │   │   │   ├── __init__.py
  │   │   │   ├── smolvlm.py
  │   │   │   ├── internvl.py
  │   │   │   └── tinyllava.py
  │   │   └── provider_models/
  │   │       ├── __init__.py
```