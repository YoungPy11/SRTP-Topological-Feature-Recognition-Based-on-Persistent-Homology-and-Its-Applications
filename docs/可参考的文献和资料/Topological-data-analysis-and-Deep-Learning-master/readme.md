## Introduction

Topological Data Analysis (TDA) is a powerful approach that reveals the underlying topological structures of data, encompassing both the physical topologies of real-world systems and the abstract topologies of high-dimensional datasets. Recently, TDA has sparked significant interest across multiple research directions, including its integration with Graph Neural Networks (GNNs), advancements in latent space representations—along with their extensions to multimodal models and knowledge distillation—and its applications in medical image processing. In this blog post, we’ll introduce high-quality recent TDA papers spanning these diverse areas, highlighting the innovative ways TDA is shaping modern data science.



<table>
<tr><td colspan="2"><a href="#surveys-tools" style="color:#B22222">1. Surveys, Tools</a></td></tr>
<tr>
    <td> <a href="#surveys">1.1 Surveys</a></td>
    <td> <a href="#code-of-tools">1.2 Code of Tools</a></td>
</tr>
<tr>
    <td> <a href="#books">1.3 Books</a></td>
    <td></td>
</tr>
<tr><td colspan="2"><a href="#application" style="color:#B22222">2. Application</a></td></tr>
<tr>
    <td> <a href="#tda-in-hyperbolic-representations">2.1 TDA in Hyperbolic Representations</a></td>
    <td> <a href="#data-representation">2.2 Data Representation</a></td>
</tr>
<tr>
    <td> <a href="#tda-in-multimodal">2.3 TDA in Multimodal</a></td>
    <td> <a href="#medical-image-and-image-segmentation">2.4 Medical Image and Image Segmentation</a></td>
</tr>
<tr>
    <td> <a href="#tda-in-knowledge-distillation">2.5 TDA in Knowledge Distillation</a></td>
    <td> <a href="#gnn-graph-neural-networks">2.6 GNN (Graph Neural Networks)</a></td>
</tr>
<tr>
    <td> <a href="#tda-tools-and-methods">2.7 TDA Tools and Methods</a></td>
    <td> <a href="#tda-in-bio-informatics">2.8 TDA in Bio-Informatics</a></td>
</tr>
<tr>
    <td> <a href="#tda-in-neural-networks-analysis">2.9 TDA in Neural Networks Analysis</a></td>
    <td> <a href="#tda-in-other-fields-of-application">2.10 TDA in Other Fields of Application</a></td>
</tr>
</table>

## Surveys, Tools, Books
<a name="surveys-tools"></a>

### Surveys
<a name="surveys"></a>

1. [A survey of topological machine learning methods](https://www.frontiersin.org/articles/10.3389/frai.2021.681108/full) - **Frontiers in AI, 2021**  
   *Felix Hensel, Michael Moor, Bastian Rieck*

### Code of Tools
<a name="code-of-tools"></a>

1. [Giotto-TDA](https://github.com/giotto-ai/giotto-tda)

2. [Gudhi](https://github.com/GUDHI/gudhi-devel)

3. [A topology layer for machine learning](http://proceedings.mlr.press/v108/gabrielsson20a.html) - **AISTATS, 2020**  
   *Rickard Bråel Gabrielsson, Bradley J. Nelson, Anjan Dwaraknath, Primoz Skraba*

4. [Pllay: Efficient topological layer based on persistent landscapes](https://proceedings.neurips.cc/paper_files/paper/2020/hash/b803a9254688e259cde2ec0361c8abe4-Abstract.html) - **NeurIPS, 2020**  
   *Kwangho Kim, Jisu Kim, Manzil Zaheer, Joon Kim, Frédéric Chazal, Larry Wasserman*

5. [GPU-accelerated computation of Vietoris-Rips persistence barcodes](http://arxiv.org/abs/2003.07989) - **Symposium on Computational Geometry, 2020**  
   *Simon Zhang, Mengbai Xiao, Hao Wang Kim, Jae-Hun Jung*

### Books

1. [Computational Topology: An Introduction](https://books.google.co.jp/books?hl=zh-CN&lr=&id=MDXa6gFRZuIC&oi=fnd&pg=PR11&dq=++Edelsbrunner&ots=CTTr7bWGoz&sig=PSIgWcXNqns1OEN8vqpyXbSn96E&redir_esc=y#v=onepage&q=Edelsbrunner&f=false) - 2010 \
   *Herbert Edelsbrunner, John Harer*

2. [Geometry and Topology for Mesh Generation](https://books.google.co.jp/books?hl=zh-CN&lr=&id=v6BybEYVGqYC&oi=fnd&pg=PP15&dq=++Edelsbrunner&ots=nwurY_Jdlg&sig=4cYx-AJzld_YfddtX6-Q1e5tat0&redir_esc=y#v=onepage&q=Edelsbrunner&f=false) - 2001 \
   *Herbert Edelsbrunner*

## Application
<a name="application"></a>


### TDA in Hyperbolic Representations
1. [Learning Hyperbolic Representations of Topological Features](https://arxiv.org/abs/2103.09273) - **ICLR, 2021** \
   *Panagiotis Kyriakis, Iordanis Fostiropoulos, Paul Bogdan*


### Data Representation
<a name="data-representation"></a>


1. [Representation topology divergence: A method for comparing neural network representations](http://arxiv.org/abs/2201.00058) - **ICML, 2022**  
   *Serguei Barannikov, Ilya Trofimov, Nikita Balabin, Evgeny Burnaev*

2. [Mapping the Multiverse of Latent Representations](http://arxiv.org/abs/2402.01514) - **ICML, 2024**  
   *Jeremy Wayland, Corinna Coupette, Bastian Rieck*

3. [Topological autoencoders](http://proceedings.mlr.press/v119/moor20a.html?ref=https://githubhelp.com) - **ICML, 2020**  
   *Michael Moor, Max Horn, Bastian Rieck, Karsten Borgwardt*

4. [Connectivity-optimized representation learning via persistent homology](http://proceedings.mlr.press/v97/hofer19a.html) - **ICML, 2019**  
   *Christoph Hofer, Roland Kwitt, Marc Niethammer, Mandar Dixit*

5. [Evaluating the disentanglement of deep generative models through manifold topology](http://arxiv.org/abs/2006.03680) - **ICLR, 2021**  
   *Sharon Zhou, Eric Zelikman, Fred Lu, Andrew Y. Ng, Gunnar Carlsson, Stefano Ermon*

6. [Manifold topology divergence: A framework for comparing data manifolds](https://proceedings.neurips.cc/paper/2021/hash/3bc31a430954d8326605fc690ed22f4d-Abstract.html) - **NeurIPS, 2021**  
   *Serguei Barannikov, Ilya Trofimov, Grigorii Sotnikov, Ekaterina Trimbach, Alexander Korotin, Alexander Filippov, Evgeny Burnaev*

7. [Learning topology-preserving data representations](http://arxiv.org/abs/2302.00136) - **ICLR, 2023**  
   *Ilya Trofimov, Daniil Cherniavskii, Eduard Tulchinskii, Nikita Balabin, Evgeny Burnaev, Serguei Barannikov*

8. [Disentanglement learning via topology](http://arxiv.org/abs/2308.12696) - **ICML, 2024**  
   *Nikita Balabin, Daria Voronkova, Ilya Trofimov, Evgeny Burnaev, Serguei Barannikov*

### TDA in Multimodal
<a name="tda-in-multimodal"></a>

1. [Homology consistency constrained efficient tuning for vision-language models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a9338cd6e092ff1f96c3749b08cdc537-Abstract-Conference.html) - **NeurIPS, 2024**  
   *Huatian Zhang, Lei Zhang, Yongdong Zhang, Zhendong Mao*

### Medical Image and Image Segmentation
<a name="medical-image-and-image-segmentation"></a>

1. [Topology preserving compositionality for robust medical image segmentation](https://openaccess.thecvf.com/content/CVPR2023W/TAG-PRA/html/Santhirasekaram_Topology_Preserving_Compositionality_for_Robust_Medical_Image_Segmentation_CVPRW_2023_paper.html) - **CVPR, 2023**  
   *Ainkaran Santhirasekaram, Mathias Winkler, Andrea Rockall, Ben Glocker*

2. [A topological loss function for deep-learning based image segmentation using persistent homology](https://ieeexplore.ieee.org/abstract/document/9186664/) - **IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020**  
   *James R. Clough, Nicholas Byrne, Ilkay Oksuz, Veronika A. Zimmer, Julia A. Schnabel, Andrew P. King*

3. [Topology-preserving deep image segmentation](https://proceedings.neurips.cc/paper/2019/hash/2d95666e2649fcfc6e3af75e09f5adb9-Abstract.html) - **NeurIPS, 2019**  
   *Xiaoling Hu, Fuxin Li, Dimitris Samaras, Chao Chen*

4. [Uncovering the topology of time-varying fMRI data using cubical persistence](https://proceedings.neurips.cc/paper/2020/hash/4d771504ddcd28037b4199740df767e6-Abstract.html) - **NeurIPS, 2020**  
   *Bastian Rieck, Tristan Yates, Christian Bock, Karsten Borgwardt, Guy Wolf, Nicholas Turk-Browne, Smita Krishnaswamy*

5. [Topologically faithful image segmentation via induced matching of persistence barcodes](https://proceedings.mlr.press/v202/stucki23a) - **ICML, 2023**  
   *Nico Stucki, Johannes C. Paetzold, Suprosanna Shit, Bjoern Menze, Ulrich Bauer*

6. [Learning probabilistic topological representations using discrete Morse theory](http://arxiv.org/abs/2206.01742) - **ICLR, 2023**  
   *Xiaoling Hu, Dimitris Samaras, Chao Chen*

### TDA in Knowledge Distillation
<a name="tda-in-knowledge-distillation"></a>

1. [Do topological characteristics help in knowledge distillation?](https://openreview.net/forum?id=2dEH0u8w0b) - **ICML, 2024**  
   *Jungeun Kim, Junwon You, Dongjin Lee, Ha Young*

2. [Topological persistence guided knowledge distillation for wearable sensor data](https://www.sciencedirect.com/science/article/pii/S0952197623019036) - **Engineering Applications of AI, 2024**  
   *Eun Som Jeon, Hongjun Choi, Ankita Shukla, Yuan Wang, Hyunglae Lee, Matthew P. Buman, Pavan Turaga*

### GNN (Graph Neural Networks)
<a name="gnn-graph-neural-networks"></a>

1. [A persistent Weisfeiler-Lehman procedure for graph classification](http://proceedings.mlr.press/v97/rieck19a.html) - **ICML, 2019**  
   *Bastian Rieck, Christian Bock, Karsten Borgwardt*

2. [Topological graph neural networks](http://arxiv.org/abs/2102.07835) - **ICLR, 2022**  
   *Max Horn, Edward De Brouwer, Michael Moor, Yves Moreau, Bastian Rieck, Karsten Borgwardt*

3. [Persistence enhanced graph neural network](https://proceedings.mlr.press/v108/zhao20d.html) - **AISTATS, 2020**  
   *Qi Zhao, Ze Ye, Chao Chen, Yusu Wang*

4. [Learning metrics for persistence-based summaries and applications for graph classification](https://proceedings.neurips.cc/paper/2019/hash/12780ea688a71dabc284b064add459a4-Abstract.html) - **NeurIPS, 2019**  
   *Qi Zhao, Yusu Wang*

5. [Persistent homology based graph convolution network for fine-grained 3D shape segmentation](http://openaccess.thecvf.com/content/ICCV2021/html/Wong_Persistent_Homology_Based_Graph_Convolution_Network_for_Fine-Grained_3D_Shape_ICCV_2021_paper.html) - **ICCV, 2021**  
   *Chi-Chong Wong, Chi-Man Vong*

6. [Topological data analysis in graph neural networks: Surveys and perspectives](https://ieeexplore.ieee.org/abstract/document/10826583/) - **IEEE Transactions on Neural Networks and Learning Systems, 2024**  
   *Phu Pham, Quang-Thinh Bui, Ngoc Thanh Nguyen, Robert Kozma, Philip S. Yu, Bay Vo*

7. [Link prediction with persistent homology: An interactive view](https://proceedings.mlr.press/v139/yan21b/yan21b.pdf) - **ICML, 2021**  
   *Zuoyu Yan, Tengfei Ma, Liangcai Gao, Zhi Tang, Chao Chen*

8. [Curvature filtrations for graph generative model evaluation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c710d6b4507e70c6332bee871b8d1ca5-Abstract-Conference.html) - **NeurIPS, 2023**  
   *Joshua Southern, Jeremy Wayland, Michael Bronstein, Bastian Rieck*

9. [Neural Approximation of Graph Topological Features](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d7ce06e9293c3d8e6cb3f80b4157f875-Abstract-Conference.html) - **NeurIPS, 2022** 
    *Zuoyu Yan, Tengfei Ma, Liangcai Gao, Zhi Tang, Yusu Wang, Chao Chen* 

### TDA Tools and Methods
<a name="tda-tools-and-methods"></a>

1. [Optimizing persistent homology based functions](http://proceedings.mlr.press/v139/carriere21a.html) - **ICML, 2021**  
   *Mathieu Carrière, Frédéric Chazal, Marc Glisse, Yuichi Ike, Hariprasad Kannan, Yuhei Umeda*

2. [Multiparameter persistence image for topological machine learning](https://proceedings.neurips.cc/paper/2020/hash/fdff71fcab656abfbefaabecab1a7f6d-Abstract.html) - **NeurIPS, 2020**  
   *Mathieu Carrière, Andrew Blumberg*

3. [Sliced Wasserstein kernel for persistence diagrams](https://proceedings.mlr.press/v70/carriere17a.html) - **ICML, 2017**  
   *Mathieu Carrière, Marco Cuturi, Steve Oudot*

4. [Persistence paths and signature features in topological data analysis](https://ieeexplore.ieee.org/abstract/document/8567922/) - **IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018**  
   *Ilya Chevyrev, Vidit Nanda, Harald Oberhauser*

5. [A stable multi-scale kernel for topological machine learning](http://openaccess.thecvf.com/content_cvpr_2015/html/Reininghaus_A_Stable_Multi-Scale_2015_CVPR_paper.html) - **CVPR, 2015**  
   *Jan Reininghaus, Stefan Huber, Ulrich Bauer, Roland Kwitt*

6. [PI-Net: A Deep Learning Approach to Extract Topological Persistence Images](https://ieeexplore.ieee.org/document/9151070/) - **CVPR Workshops, 2020**  
   *Anirudh Som, Hongjun Choi, Karthikeyan Natesan Ramamurthy, Matthew P. Buman, Pavan Turaga*

### TDA in Bio-Informatics
<a name="tda-in-bio-informatics"></a>

1. [Representation of molecular structures with persistent homology for machine learning applications in chemistry](https://www.nature.com/articles/s41467-020-17035-5) - **Nature Communications, 2020**  \
   *Jacob Townsend, Cassie Putman Micucci, John H. Hymel, Vas Angkor Wat, Vasileios Maroulas, Konstantinos D. Vogiatzis*

2. [Improving Self-supervised Molecular Representation Learning using Persistent Homology](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6b555e8552240d6dfe0767146c9ebf36-Abstract-Conference.html) - **NeurIPS, 2023** \
   *Yuankai Luo, Lei Shi, Veronika Thost*

3. [ToDD: Topological Compound Fingerprinting in Computer-Aided Drug Discovery](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b31f6d65f2584b3c4347148db36fe07f-Abstract-Conference.html) - **NeurIPS, 2022** \
   *Andaç Demir, Baris Coskunuzer, Yulia Gel, Ignacio Segovia-Dominguez, Yuzhou Chen, Bulent Kiziltan*

### TDA in Neural Networks Analysis
<a name="tda-in-neural-networks-analysis"></a>

1. [Neural persistence: A complexity measure for deep neural networks using algebraic topology](http://arxiv.org/abs/1812.09764) - **ICLR, 2019**  
   *Bastian Rieck, Matteo Togninalli, Christian Bock, Michael Moor, Max Horn, Thomas Gumbsch, Karsten Borgwardt*

### TDA in Other Fields of Application
<a name="tda-in-other-fields-of-application"></a>

1. [Deep learning with topological signatures](https://proceedings.neurips.cc/paper_files/paper/2017/hash/883e881bb4d22a7add958f2d6b052c9f-Abstract.html) - **NeurIPS, 2017**  
   *Christoph Hofer, Roland Kwitt, Marc Niethammer, Andreas Uhl*

2. [Learning representations of persistence barcodes](http://www.jmlr.org/papers/v20/18-358.html) - **Journal of Machine Learning Research, 2019**  
   *Christoph D. Hofer, Roland Kwitt, Marc Niethammer*