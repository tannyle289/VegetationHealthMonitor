Hi, welcome to my project, this is my personal project called: Vegetation Health Monitor using some SoTA techniques and frameworks in AI, that is made to practice my skills in those fields.

For initial setup, please look at [README_SETUP](./README_SETUP.md)

I sketched this project into 4 phases, which might be changed constantly alongside the implementation, personally I don't even know if any phase of this project is doable for me, but I will learn of course and try to keep this file up to date. Because of that reason, this file will be finished last so the tone, content are not finalized like in README_SETUP :) Please understand

Also, I would be more than grateful and open to receive all comments, feedbacks or contributions!! 🙏 

# Idea: 
Create an end-to-end, deployable system that automatically detects urban trees in RGB imagery, fuses per-tree LiDAR structure across time, and outputs a calibrated health score + uncertainty and actionable recommendations via a small LLM agent, while demonstrating real-time, edge-capable inference (ONNX / TensorRT) on a Jetson-sim environment.

Sounds ambitious huh, I know... I will try my best.

## Phase 1: Image detector for tree + tracking
- [x] Literature review and paper/project researches for inferences
- [x] Exploring SoTA tree/tree health datasets
- [x] Training a segmenter on the Urban Street Tree dataset
- [ ] Using Ultralytics tracking with BYTETrack over a video or ordered frame folder to get persistent IDs per tree.

**Expected Outcome:** per‑frame masks and per‑tree track IDs, the idea is I want these IDs become the index to aggregate features and measurements for each tree over time in later phases.

## Phase 2: LiDAR/3D structure
- [ ] Use point clouds individual trees and extract unique characteristics like DBH, height, canopy volume/density, shape, ...; store them accordingly with timestamps and geo keys to manage growth trends.
- [ ] Visualize the tree in point cloud

**Expected Outcome:** Tree visualization and LiDAR metrics.

## Phase 3: Health estimation
- [ ] Build a simple health score from images (color deviation, canopy density, ...) also combined with LiDAR cues in Phase 2 for a more robust condition estimate
- [ ] Train a temporal aggregator (RNN/Transformer) to track changes over time per tree, as the result estimate confidence or uncertainty
- [ ] Train small regressor/classifier on fused features to predict health score 

**Expected Outcome:** a per‑tree health score per timestamp, enabling flags like stable/improving/declining.


## Phase 4:  LangChain/LangGraph multi‑agent on the edge (IDK which will I use yet)
- [ ] Add a small agent service that take all above inputs (vision API, LiDAR, ...) to provide reasoning and return  decisions like "urgent prune," "enlarge pit 30 cm," or "plant drought‑tolerant species," ...

## Tech Summary:

To sum up, the project might involve:

Tree Image/Frame → Segmenter (YOLOv11) → LiDAR/Point cloud → Temporal (RNN / GRU / LSTM / Transformer / ViT-based) → Fusion + Health Prediction (MLP, small CNN) → Reasoning and Decision-Making (text, LLM)

