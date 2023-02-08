Here we describe briefly the maths behind Scyan.

## Notations

| Symbol      | Belongs to | Description                          |
| ----------- | -----------|------------------------------------ |
| $P$ | $\mathbb{N}^*$ | Number of populations  |
| $N$ | $\mathbb{N}^*$ | Number of cells |
| $M$ | $\mathbb{N}^*$ | Number of markers |
| $M_{c}$ | $\mathbb{N}^*$ | Number of covariates per cell |
| $\pmb{\pi} = (\pi_z)_{1 \leq z \leq P}$ | $]0, 1[^P$ | Population size ratios (i.e., $\sum_z \pi_z = 1$) |
| $\pmb{\rho}$ | $(\mathbb{R} \cup \{NA\})^{P \times M}$ | Knowledge table. For a population $z$ and a marker $m$, the value $\rho_{z,m}$ describes the expected expression of $m$ by population $z$ (typically, -1 for negative, 1 for positive, NA when not known)| 
| $\pmb{x_i}$ | $\mathbb{R}^M$ | Each $\pmb{x_i}$ is the pmbtor of preprocessed marker expression of the cell $i \in [1\cdots N]$|
| $\pmb{c_i}$ | $\mathbb{R}^{M_c}$ | Each $\pmb{c_i}$ is the pmbtor of covariates associated to the cell $i \in [1\cdots N]$|

## Generative process

We make the assumption that the cell expressions comes from this generative process, where $f_{\pmb{\phi}}$ is the normalizing flow detailed in the next section.

\[
    Z \sim Categorical(\pmb{\pi})
\]

\[\pmb{E} \; | \; Z = (e_m)_{1 \leq m \leq M} \mbox{, where }
        \left\{
            \begin{array}{ll}
                e_m = \rho_{Z,m} & \mbox{if }\rho_{Z,m} \neq \mbox{NA} \\
                e_m \sim \mathcal{U}([-1, 1]) & \mbox{otherwise,}
            \end{array}
        \right.
\]

\[
    \pmb{H} \sim \mathcal{N}(\pmb{0}, \sigma \mathbb{\pmb{I_M}})
\]

\[
    \pmb{U} = \pmb{E} + \pmb{H}
\]

\[
    \pmb{X} = f_{\pmb{\phi}}^{-1}(\pmb{U})
\]

The latent distribution $\pmb{U}$ is defined on $\mathbb{R}^M$, i.e. the same space as the original marker expressions.

## Normalizing flow

The normalizing flow is a stack of multiple coupling layers: $f_{\pmb{\phi}} := f^{(L)} \circ f^{(L-1)} \circ \dots \circ f^{(1)}$ with $L$ the number of coupling layers. Each coupling layer $f^{(i)}: (\pmb{x}, \pmb{c}) \mapsto \pmb{y}$ splits both $\pmb{x}$ and $\pmb{y}$ into two components $(\pmb{x^{(1)}}, \pmb{x^{(2)}}), (\pmb{y^{(1)}}, \pmb{y^{(2)}})$ on which distinct transformations are applied. We propose below an extension of the traditional coupling layer to integrate covariates $\pmb{c}$:

\[
    \begin{cases}
      \pmb{y^{(1)}} = \pmb{x^{(1)}}\\
      \pmb{y^{(2)}} = \pmb{x^{(2)}} \odot exp\Big(s([\pmb{x^{(1)}}; \pmb{c}])\Big) + t([\pmb{x^{(1)}}; \pmb{c}]).
    \end{cases}  
\]

On the equation above, $s$ and $t$ are Multi-Layer-Perceptrons (MLPs).

!!! note
    Each coupling layer is inversible, so is $f_{\pmb{\phi}}$. Also, the log-determinent of the jacobian of each layer is easy to compute, which is crucial to compute the loss function below.

## Loss
We optimize by stochastic gradient descent the Kullback–Leibler (KL) divergence defined below. More details leading to this expression and how to compute it can be found in the article methods section.

\[
    \mathcal{L}_{KL}(\pmb{\theta}) = - \sum_{1 \leq i \leq N} \bigg[ log \: \Big( p_U(f_{\pmb{\phi}}(\pmb{x_i}, \pmb{c_i}); \pmb{\pi}) \Big) + log \;  \Big| det \frac{\partial f_{\pmb{\phi}}(\pmb{x_i}, \pmb{c_i})}{\partial \pmb{x}^T} \Big| \bigg].
\]

We optimize the loss on mini-batches of cells using the Adam optimizer.

## Correcting batch effect

When the batches are provided into the covariates, the normalizing flow with naturally learn to align the latent representations of the multiple different batches. After model training, we can choose a batch as being the reference and its corresponding covariates $\pmb{c}_{ref}$. Then, for a cell $\pmb{x}$ with covariates $\pmb{c} \neq \pmb{c_{ref}}$, its batch-effect corrected expressions are, $\tilde{\pmb{x}} = f_{\pmb{\phi}}^{-1}\Big(f_{\pmb{\phi}}(\pmb{x}, \pmb{c}), \pmb{c_{ref}}\Big)$.

!!! note
    In this manner, we get expressions $\tilde{\pmb{x}}$ as if $\pmb{x}$ were cell expressions from the reference batch. Applying $f_{\pmb{\phi}}$ removes the batch-effect, and applying $f_{\pmb{\phi}}^{-1}$ recreates expressions that look like expressions of the reference batch.