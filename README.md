# GRASP: Gradient-based Retention of Adaptive Singular Parameters

## Abstract

Layer removal is an effective technique for compressing large language models (LLMs) by reducing redundancy and improving inference efficiency. However, indiscriminate pruning disrupts representation stability, leading to performance degradation. We propose \textbf{GRASP} (Gradient-based Retention of Adaptive Singular Parameters), which preserves representation-critical singular values to mitigate these effects. Unlike direct layer removal, GRASP leverages gradient-based attribution on a syntax and semantics-rich dataset to guide the selection of representation-critical singular values. By selectively applying singular value decomposition (SVD) to affected layers, GRASP achieves efficient compression while maintaining representation stability with minimal overhead. Experiments across multiple LLMs show that GRASP consistently outperforms existing compression methods in perplexity and downstream task performance.

![GRASP](assets\GRASP.png)