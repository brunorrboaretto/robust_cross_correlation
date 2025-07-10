# cross_correlation_birds_song

This repository hosts the data, code, and resources associated with the paper titled:

**"A robust framework for linking neural activity to vocal output in birdsong"**

### Authors:
- **Fiamma L. Leites***  
  Universidad de Buenos Aires, Facultad de Ciencias Exactas y Naturales, Departamento de Física, Argentina  
  CONICET – Universidad de Buenos Aires, Instituto de Física Interdisciplinaria y Aplicada (INFINA), Argentina  

- **Bruno R. R. Boaretto***  
  Institute of Science and Technology, Universidade Federal de São Paulo, São José dos Campos, Brazil  
  Department of Physics, Universitat Politècnica de Catalunya, Terrassa, Spain  

- **Cristina Masoller**  
  Department of Physics, Universitat Politècnica de Catalunya, Terrassa, Spain  

- **Ana Amador**  
  Universidad de Buenos Aires, Facultad de Ciencias Exactas y Naturales, Departamento de Física, Argentina  
  CONICET – Universidad de Buenos Aires, Instituto de Física Interdisciplinaria e Aplicada (INFINA), Argentina  

> *These authors contributed equally to this work.
<!-- a normal html comment //XXXXX INCLUDE THE INFORMATION OF THE PAPER HERE.... DOI EDICTOS PICKS BLA BLA BAL FOCUS ISSUE '  <code>DOI: 10.1063/5.0193967</code>

 -->
<!--Volume #:	34-->
<!--Issue #:	4-->
<!--Issue:	2024-04-30-->
 
## Paper Overview

This repository presents a method to quantify temporal correlations between neural activity and motor output during canary song production. We analyze multiunit activity (MUA) recorded extracellularly from the brain, alongside the acoustic envelope (ENV) of the song. The approach uses cross-correlation functions, $C_{XY}(\tau)$ and $C_{YX}(\tau)$, computed in short sliding windows, to identify characteristic time lags ($\tau^*$) that indicate whether neural activity precedes or follows vocal output. Statistical significance is assessed using block-shuffled and surrogate data. Our results show that, even under nonstationary and noisy conditions, the method reliably detects meaningful correlations and time lags between MUA and ENV. This framework is broadly applicable to other neural-motor systems where short, complex, and rhythmic signals are involved.

## Repository Contents

This repository includes the following components:

- **Datasets:** Six `.dat` files: three contain a few seconds of neural multiunit activity (MUA) recorded extracellularly from the brain, and the other three contain the corresponding acoustic envelope (ENV) of the canary song.

- **Code:** Python implementation for computing the cross-correlation functions with time delays between MUA and ENV signals. The code also includes statistical validation using surrogate and block-shuffled data to assess the significance of the observed correlations.
   
## Dataset

We analyze experimental recordings from adult male canaries (*Serinus canaria*), captured during song production. Two types of time series were recorded **simultaneously**:

- **Multiunit Activity (MUA):** Extracellular recordings of population-level neural activity from the HVC (proper name), a premotor nucleus involved in vocal behavior.
- **Envelope (ENV):** The amplitude envelope of the canary song audio signal.

Recordings were obtained from two individuals and are labeled using the following convention:

- `Rona` or `Vioama` — Name of the bird
- A numerical ID — Stereotaxic coordinate of the electrode (e.g., `404`, `978`, `310`)

### Files Included

| Signal Type | Subject | File Name              |
|-------------|---------|------------------------|
| MUA         | Rona\_404   | `data_MUA_404.dat`      |
| ENV         | Rona\_404   | `data_ENV_404.dat`      |
| MUA         | Vioama\_978 | `data_MUA_978.dat`      |
| ENV         | Vioama\_978 | `data_ENV_978.dat`      |
| MUA         | Rona\_310   | `data_MUA_310.dat`      |
| ENV         | Rona\_310   | `data_ENV_310.dat`      |

Each `.dat` file contains a long time series composed of thousands of milliseconds used for cross-correlation analysis between MUA and ENV signals.

**Note:** File suffixes (`404`, `978`, `310`) match the stereotaxic coordinates for easy pairing of MUA and ENV data.

## Python libraries:

- **NumPy**: Facilitates efficient handling and manipulation of large multi-dimensional arrays and provides a wide range of mathematical functions for numerical computations in Python;
- **matplotlib.pyplot** is a Python library commonly used for creating visualizations and plots, providing a high-level interface for generating a wide range of graphs and charts;
- **scipy.signal**: We use functions from the `scipy.signal` module (part of the SciPy library) to perform signal processing tasks. In particular, the `find_peaks` function is employed to identify prominent peaks in the cross-correlation functions.
- **IAAFT Surrogate Module**: We use a Python implementation of the Iterative Amplitude Adjusted Fourier Transform (IAAFT) method, originally developed by [Bedartha Goswami], to generate surrogate time series that preserve both the amplitude distribution and power spectrum of the original signal. This library
  
*Note: The IAAFT library should be downloaded from <code>https://github.com/mlcs/iaaft</code> and has to be in the same directory as the code;*

## Code

For each sample (`404`, `978`, and `310`), the analysis follows two main steps. First, run the script `<code>analysis_cross_corr.py</code>` to:

- Load the MUA and ENV data;
- Define hyperparameters such as window size, window step, maximum lag, prominence threshold, and tolerance threshold;
- Segment the signals into sliding windows;
- Generate surrogate datasets (block shuffled and IAAFT surrogates);
- Compute the windowed cross-correlation for each segment;
- Extract the peak cross-correlation value and its associated delay ($C^*$ and $τ^*$), or mark the segment as undefined if the significance criteria are not met.

Once this script has been executed for all three samples, three result files will be created. Then, execute `<code>plot_cross_corr.py</code>` to generate the visualizations corresponding to Figs. 4, 5, and 6 of the manuscript.

### Suggested execution

```bash
python3 analysis_cross_corr.py 0;
python3 analysis_cross_corr.py 1;
python3 analysis_cross_corr.py 2;
python3 plot_cross_corr.py;
```
*Note: The IAAFT surrogate generation is performed for each window, which can make the analysis computationally intensive and time-consuming (several minutes per file).*

## Citation

If you find this work helpful for your research, please consider reading and citing:

- Leites, F. L., Boaretto, B. R. R., Masoller, C., & Amador, A., "A robust framework for linking neural activity to vocal output in birdsong". *Under review*.



--------------------------------------------------------------------------------------

Thank you for your interest in our research! </br>
We hope this repository serves as a valuable resource for spike timing analysis in chaotic lasers and data analysis in general.</br>

Sincerely,</br>
Bruno R. R. Boaretto.


