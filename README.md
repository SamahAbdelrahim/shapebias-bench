# shapebias-bench
A benchmark for evaluating the shape bias in CNN and VLM 

two folders:
stimuli_pipe to create stimuli 
evaluation and model pipeline 

Stimuli onboarding:
- Start with `STIMULI_GUIDE.md` (repo root/stimuli_pipe)
- Then read:
  - `stimuli_pipe/stimuli_repro_bundle/README.md`
  - `stimuli_pipe/stimuli_repro_bundle/STIMULI_GUIDE.md`
- For benchmark use, point to `stimuli_pipe/stimuli_per_stl_packages` only.


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
  ├── stimuli_pipe/
```