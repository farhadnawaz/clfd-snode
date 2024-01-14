# Stable Continual Learning from Demonstration
This repository contains the code and datasets for our paper **Scalable and Efﬁcient Continual Learning from Demonstration via a Hypernetwork-generated Stable Dynamics Model** ([preprint](https://arxiv.org/abs/2311.03600)).

We propose a novel hypernetwork-based approach for stable continual learning from demonstration, enabling a robot to sequentially acquire multiple trajectory-based stable motion skills with a single hypernetwork without retraining on previous demonstrations. The stability in learned trajectories not only prevents divergence in motion but also greatly enhances continual learning performance, particularly in our most size-efficient model. We propose an efficient hypernetwork training method and provide new datasets for the LfD community. Evaluations on various benchmarks, including real-world robotic manipulation tasks, validate our approach.

<p style="text-align:center">
  <img src="images/clfd_snode_pred_all_1024.gif" width=" 800" /> 
  <figcaption>The robot is able to reproduce any task after continually learning the 9 realistic tasks of the RoboTasks9 dataset (each task involves changing positions and orientations).</figcaption>
</p>

Here is a short overview of our approach:

https://github.com/sayantanauddy/clfd-snode/assets/10401716/1958249e-5cab-4ae3-887b-2a7c6f2da0b6

