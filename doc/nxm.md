# Multi-channel, multi-template event reconstruction

In a multi-channel cryogenic particle detector based on TES sensor, there are usually two challenges. First, pulse shape varies. Second, there is noise correlation between channels. We develope a event reconstruction workflow to adress these two issues. 

## 1. Single-channel, multi-template event reconstruction

We first deal with pulse shape variation in a single channel. Even for monoenergetic event, the pulse shape in one TES channel could vary based on the location of that event. Instead of using a multi-falltime analytical fit, we want to use a multi-template matched filter for the robustness near threshold.

Multi-template matched filter is well described in many textbooks and we will skip the introduction. However, simply using PCA components as templates will give the wrong reconstruction. Cares must be taken in modeling the pulse and generating templates.

### 1.1 Model pulse using PCA components

Let $\mathbf{x}\in\mathbb R^M$ be a sampled pulse, and let the additive detector noise have zero mean and positive-definite covariance $N$. Define $W=N^{-1}$.

A physically useful model for approximately **scale-invariant** shape fluctuations is

$$
\boxed{
\mathbf x(A,c_1,\ldots, c_k)=A\left(\mathbf s_0+
\sum_k c_k\mathbf s_k\right)
+\boldsymbol\epsilon, \qquad  c_k\sim\mathcal N(0,\lambda_k)
}
$$


where:
- $\mathbf s_0$ is the mean pulse template.
- $A$ is the overall pulse amplitude.
- $\mathbf s_k$ is the $k$-th PCA direction, from $\texttt{pca.components\_[k]}$. The PCA directions satisfy $\|\mathbf s_k\|_2=1.$
- $\lambda_k$ is the variance from $\texttt{pca.explained\_variance\_[k]}$.
- $c_k$ is the fitted coefficient for that direction.
- $\boldsymbol\epsilon$ is measurement noise.


For numerical stability, we can parameterize the fit coefficients  $c_k$ as
$$
c_k=\sqrt{\lambda_k}\,z_k,
\qquad z_k\sim\mathcal N(0,1).
$$

and define the template matrix
$$
\mathbf B=\mathbf (\mathbf s_1\sqrt{\lambda_1},\ldots,\mathbf s_k \sqrt{\lambda_k}).
$$

The model can be written as

$$
\boxed{
\mathbf x(A,\mathbf z)=A\left(\mathbf s_0+
\mathbf B \mathbf z\right)
+\boldsymbol\epsilon, \qquad  z_k\sim\mathcal N(0,1)
}
$$


### 1.2. MAP objective and solution

The negative twice log posterior is

$$
\boxed{
Q(A,\mathbf z)
=\bigl[\mathbf x-A(\mathbf s_0+\mathbf B \mathbf z)\bigr]^T
W\bigl[\mathbf x-A(\mathbf s_0+\mathbf B \mathbf z)\bigr]
+\mathbf z^T\mathbf z.
}
$$
This is almost the same with a multi-template matched filter case with two differencies. There is an additional term, whichis the regularization induced by the PCA prior. Plus, model parameters are no longer quadratic.

Unlike the simple multi-template matched filter, there is no analytical solution to all model parameters $(A,\mathbf z)$. If an amplitude $\mathrm A$ is given, there is the exact conditional solution for z:

$$
\boxed{
\widehat{\mathbf z}(A)
=\left(A^2\mathbf{B}^TW\mathbf{B}+C^{-1}\right)^{-1}
A \mathbf{B}^TW(\mathbf x-A\mathbf s_0).
}
$$

### 1.3 


