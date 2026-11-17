# Projection quality vs N — jinaclip-512-1m

36 threads · seeds [0, 1, 2] · mean ± sd over seeds. **combined** = mean of local and global after min-max scaling each across the methods at that N (relative: compare within one N only). Local scores use 10,000 anchor points above N = 100,000, every point otherwise.

![Projection quality](figures/jinaclip-512-1m-quality-map.png)

## N = 10,000

| algorithm | combined | local (trust.) | global (pair ρ) | triplet | fit s | peak RAM GB |
| --- | --- | --- | --- | --- | --- | --- |
| mrs-dreams-fft-rinnd-default | 0.89 | 0.9008 ± 0.0013 | 0.244 ± 0.021 | 0.586 ± 0.004 | 2.2 | 0.5 |
| opentsne-fft-rinnd-default | 0.78 | 0.9070 ± 0.0020 | 0.198 ± 0.021 | 0.568 ± 0.004 | 35.0 | 0.7 |
| ncvis-plus-rinnd-default | 0.78 | 0.9119 ± 0.0009 | 0.192 ± 0.002 | 0.573 ± 0.000 | 1.0 | 0.4 |
| mrs-tsne-fft-rinnd-default | 0.78 | 0.9117 ± 0.0011 | 0.192 ± 0.027 | 0.570 ± 0.005 | 2.4 | 0.6 |
| ncvis-rinnd-default | 0.57 | 0.8965 ± 0.0018 | 0.126 ± 0.021 | 0.552 ± 0.005 | 0.5 | 0.2 |
| pca | 0.50 | 0.6443 ± 0.0000 | 0.278 ± 0.000 | 0.601 ± 0.000 | 0.1 | 0.2 |
| umap-learn-rinnd-default | 0.48 | 0.8522 ± 0.0031 | 0.122 ± 0.014 | 0.554 ± 0.003 | 1.8 | 0.5 |
| mrs-umap-rinnd-default | 0.40 | 0.8160 ± 0.0041 | 0.115 ± 0.006 | 0.550 ± 0.003 | 0.5 | 0.1 |
| mrs-pacmap-rinnd-default | 0.31 | 0.8122 ± 0.0023 | 0.087 ± 0.009 | 0.549 ± 0.002 | 3.5 | 0.5 |

## N = 100,000

| algorithm | combined | local (trust.) | global (pair ρ) | triplet | fit s | peak RAM GB |
| --- | --- | --- | --- | --- | --- | --- |
| mrs-tsne-fft-rinnd-default | 0.91 | 0.9572 ± 0.0006 | 0.214 ± 0.014 | 0.578 ± 0.004 | 15.4 | 1.6 |
| opentsne-fft-rinnd-default | 0.87 | 0.9582 ± 0.0002 | 0.202 ± 0.005 | 0.574 ± 0.002 | 82.9 | 1.6 |
| mrs-dreams-fft-rinnd-default | 0.77 | 0.9352 ± 0.0010 | 0.184 ± 0.013 | 0.570 ± 0.006 | 15.6 | 1.5 |
| ncvis-plus-rinnd-default | 0.76 | 0.9441 ± 0.0000 | 0.176 ± 0.002 | 0.567 ± 0.001 | 3.8 | 0.7 |
| ncvis-rinnd-default | 0.52 | 0.9222 ± 0.0007 | 0.109 ± 0.022 | 0.546 ± 0.005 | 4.4 | 0.4 |
| pca | 0.50 | 0.6442 ± 0.0000 | 0.243 ± 0.000 | 0.597 ± 0.000 | 0.4 | 0.5 |
| umap-learn-rinnd-default | 0.49 | 0.8816 ± 0.0003 | 0.121 ± 0.009 | 0.551 ± 0.001 | 7.3 | 0.8 |
| mrs-umap-rinnd-default | 0.42 | 0.8525 ± 0.0010 | 0.113 ± 0.015 | 0.549 ± 0.004 | 3.3 | 0.4 |
| mrs-pacmap-rinnd-default | 0.30 | 0.8330 ± 0.0002 | 0.086 ± 0.003 | 0.547 ± 0.001 | 24.6 | 2.2 |

## N = 512,000

| algorithm | combined | local (trust.) | global (pair ρ) | triplet | fit s | peak RAM GB |
| --- | --- | --- | --- | --- | --- | --- |
| mrs-tsne-fft-rinnd-default | 0.86 | 0.9683 ± 0.0006 | 0.188 ± 0.018 | 0.569 ± 0.005 | 73.2 | 5.6 |
| opentsne-fft-rinnd-default | 0.80 | 0.9705 ± 0.0002 | 0.168 ± 0.003 | 0.564 ± 0.001 | 241.8 | 4.2 |
| ncvis-plus-rinnd-default | 0.79 | 0.9544 ± 0.0003 | 0.173 ± 0.002 | 0.565 ± 0.001 | 16.5 | 1.8 |
| mrs-dreams-fft-rinnd-default | 0.78 | 0.9394 ± 0.0015 | 0.176 ± 0.008 | 0.568 ± 0.003 | 74.9 | 5.4 |
| umap-learn-rinnd-default | 0.51 | 0.8887 ± 0.0021 | 0.114 ± 0.010 | 0.549 ± 0.003 | 43.3 | 2.3 |
| pca | 0.50 | 0.6443 ± 0.0000 | 0.234 ± 0.000 | 0.595 ± 0.000 | 1.4 | 1.7 |
| ncvis-rinnd-default | 0.43 | 0.9275 ± 0.0010 | 0.067 ± 0.009 | 0.531 ± 0.002 | 22.7 | 1.8 |
| mrs-umap-rinnd-default | 0.40 | 0.8548 ± 0.0006 | 0.093 ± 0.011 | 0.544 ± 0.004 | 18.0 | 1.8 |
| mrs-pacmap-rinnd-default | 0.38 | 0.8435 ± 0.0007 | 0.092 ± 0.001 | 0.545 ± 0.000 | 120.0 | 8.6 |
