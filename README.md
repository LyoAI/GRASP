# GRASP: Gradient-based Retention of Adaptive Singular Parameters

![GRASP](./assets/GRASP.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/yourusername/GRASP/actions)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

## Overview

Layer removal is an effective technique for compressing large language models (LLMs) by reducing redundancy and improving inference efficiency. However, indiscriminate pruning disrupts representation stability, leading to performance degradation. We propose **GRASP** (Gradient-based Retention of Adaptive Singular Parameters), which preserves representation-critical singular values to mitigate these effects. Unlike direct layer removal, GRASP leverages gradient-based attribution on a syntax and semantics-rich dataset to guide the selection of representation-critical singular values. By selectively applying singular value decomposition (SVD) to affected layers, GRASP achieves efficient compression while maintaining representation stability with minimal overhead. Experiments across multiple LLMs show that GRASP consistently outperforms existing compression methods in perplexity and downstream task performance.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GRASP.git
   cd GRASP
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Adjust the parameters in `scripts/params_script.sh` to fit your needs.

2. Run the GRASP program:
   ```bash
   bash scripts/run_grasp.sh
   ```

### Evaluation

To evaluate the model, use the following script:
```bash
bash scripts/run_evaluate.sh
```

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the contributors and the open-source community.