# Diffusion Meets Numerical Methods

## 🎯 Objective:
To investigate the effect of **different numerical methods for solving DDIMs** on the image generation process in diffusion models.  
Classical DDIMs use Euler's method - we replace it with **Runge-Kutta**, **adaptive** and other integrators to improve the **quality**, **stability** or **speed** of generation.

The study compares different integrators:
- `ddim_step`
- `euler_step`
- `midpoint_step`
- `rk4_step`
- `heun_step`

In more detail, each approach has the following mathematical description: 

### DDIM step
Simplified method for diffusion models, skips steps to accelerate generation using predicted noise.
$$
x_{t-1} = \sqrt{\alpha_{t-1}} \left( \frac{x_t - \sqrt{1-\alpha_t} \epsilon_\theta(x_t, t)}{\sqrt{\alpha_t}} \right) + \sqrt{1-\alpha_{t-1}} \cdot \epsilon_\theta(x_t, t), 
$$
where $\alpha_{t}$ - diffusion coefficents, $\epsilon_\theta$ - predicted noise.

### Euler step
The simplest method: takes one step forward using the current derivative. Fast, but inaccurate.
$$
x_{t+1} = x_t + h \cdot f(x_t, t),
$$
where $h$ - step, $f(x_t, t)$ - predicted noise in diffusion process.

### Midpoint step (Midpoint method)
Improved Euler: first calculates the derivative in the middle of a step to refine the motion. More accurate, but requires two calculations.
$$
k_1 = h \cdot f(x_t, t)
$$
$$
k_2 = h \cdot f\left(x_t + \frac{k_1}{2}, t + \frac{h}{2}\right)
$$
$$
x_{t+1} = x_t + k_2
$$

### RK4 step (Runge-Kutta method of 4th order)\
Combines 4 intermediate calculations to minimise the error. Slower but more reliable.
$$
k_1 = h \cdot f(x_t, t)
$$
$$
k_2 = h \cdot f\left(x_t + \frac{k_1}{2}, t + \frac{h}{2}\right)
$$
$$
k_3 = h \cdot f\left(x_t + \frac{k_2}{2}, t + \frac{h}{2}\right)
$$
$$
k_4 = h \cdot f(x_t + k_3, t + h)
$$
$$
x_{t+1} = x_t + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$

### Heun step
Predicts a pitch like Euler, then corrects it. More accurate than Euler, but simpler than RK4.
$$
k_1 = h \cdot f(x_t, t)
$$
$$
k_2 = h \cdot f(x_t + k_1, t + h)
$$
$$
x_{t+1} = x_t + \frac{1}{2}(k_1 + k_2)
$$

## ✅ Installation
To run this project, you'll need to set up a Python environment and install the necessary dependencies.

### Prerequisites
Make sure you have Python 3.10 or higher installed.

1. Clone the repository:
```bash
git clone https://github.com/vlvink/DDIM-Integrators-Comparison.git
cd diff-solv
```

2. Install the requirements
```bash
poetry install
```

3. Setting the poetry environment
```bash
poetry shell
```

## ▶️ Running the Code (Inference mode)
### Streamlit app
If you want to run code using UI web interface, you need to configure Streamlit app:
```bash
streamlit run streamlit_page.py
```
For stopping session press the keyboard shortcut **Ctrl+C** in the terminal.

### CLI
If you want to run this code using command line of your terminal, you need to run:
```bash
python main.py
```

## ▶️ Running the Code (Training mode)
For running training process, you need to make sure you have enough GPU on a board. Reccomend to use:
- V100 x3 GPU
- ...

## 📦 Project Organization
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project
    ├── data
    │   └──       
    │   │   ├── 
    │   │   └── 
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   ├── params         <- Model's parameters in .pth files
    │   │   ├── 
    │   │   ├── 
    │   │   
    │   └── predictions    <- Model's predictions: intermediate model responses
    │
    ├── notebooks          <- Jupyter notebooks. Test code, training code
    │
    ├── src                <- Source code for use in this project
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── applications   
    │   │   ├── trainer.py
    │   │   ├── utils.py
    │   │
    │   ├── data           <- Scripts to modify data
    │   │   ├── 
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │              predictions
    │   │   ├── utils  
    │   │   │   ├── 
    │   │   │   └── 
    │   │   │
    │   │   ├── unet.py
    │   │   ├── vae.py
