# Overview

This document is used as a means to analyse the Project presentation and discuss the 5 different mandatory components we must have:

- Vision
- Global Navigation
- Motion Control
- Local Navigation
- Filtering

## Vision

## 1. Create Environment

For global navigation, the robot "should avoid obstacles without using sensors to detect them". Meaning we generate a map through our overhead camera.

## 2. Finding Best Path

The map generated above is then fed to a path planner and returns an ideal path the robot should take. It is important to note that the start and end position are allowed to change, and this will be tested (e.g. robot picked up and placed somewhere else)

## 3. Motion Control and Pose Estimation

To be able to follow the path, we must be able to accurately measure and determine the position of the robot, which will have to be aquired through some filtering technique (Bayesian, Kalman etc)

## 4. Obstacle Avoidance

While trying to follow its path, the robot must be able to avoid local obstacles that happen to appear, here the robot must make use of its sensors to percieve and adapt.

## State Machine

```mermaid

stateDiagram-v2
    [*] --> DETECT_ELEMENTS: Overhead vision camera
    DETECT_ELEMENTS --> GENERATE_MAP: vertice detection
    GENERATE_MAP --> GENERATE_GLOBAL_PLAN: A*
    GENERATE_GLOBAL_PLAN --> FOLLOW_PATH : Valid Path Generated

    FOLLOW_PATH --> FOLLOW_PATH : Obstacle Dist > Threshold
    FOLLOW_PATH --> LOCAL_AVOIDANCE : Obstacle Dist <= Threshold

    LOCAL_AVOIDANCE --> FOLLOW_PATH : Obstacle Cleared

    FOLLOW_PATH --> KIDNAPPED_STATE : Localization Anomaly Spike
    LOCAL_AVOIDANCE --> KIDNAPPED_STATE : Localization Anomaly Spike

    KIDNAPPED_STATE --> GENERATE_GLOBAL_PLAN : New Stable Pose Acquired

    FOLLOW_PATH --> [*] : Goal Reached
``
