# SVD-Based Image Compression

## MFAD Mini Project – Mathematical Foundation for AI & Data Science

### 👥 Team Members

- Rachana Y.
- Ranjani S
- Sharika Setty H
- Sannidhi G

### 📌 Project Overview

This project demonstrates **Singular Value Decomposition (SVD)** and its application to **image compression**.

SVD is an important linear algebra technique used in Artificial Intelligence, Data Science, Machine Learning, image processing, dimensionality reduction, and data compression.

In this project, an image is represented as a matrix and decomposed using SVD. The most significant singular values are retained to reconstruct an approximate version of the original image.

---

## 🎯 Objectives

- Understand the mathematical concept of SVD.
- Represent an image as a matrix.
- Decompose the image using SVD.
- Reconstruct an image using selected singular values.
- Study the relationship between compression and image quality.
- Demonstrate the application of linear algebra in AI and Data Science.

---

## 🧮 Mathematical Foundation

For a matrix **A**, Singular Value Decomposition is:

\[
A = U\Sigma V^T
\]

where:

- **A** → Original image matrix
- **U** → Left singular vectors
- **Σ** → Diagonal matrix of singular values
- **Vᵀ** → Transpose of the right singular-vector matrix

Using only the first **k** singular values gives an approximation:

\[
A_k = U_k\Sigma_kV_k^T
\]

A smaller value of **k** provides greater compression but may reduce image quality.

---

## 🔄 Workflow

```text
Input Image
     ↓
Convert Image to Matrix
     ↓
Apply SVD
     ↓
A = UΣVᵀ
     ↓
Select Top k Singular Values
     ↓
Reconstruct Image
     ↓
Compare Results
