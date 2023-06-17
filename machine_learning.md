# 1. 機械学習

機械学習のまとめを行う。数式の詳細な証明は省略し、各学習のアイデア、結論、それを実装するためのPythonのライブラリについてまとめる。

## 1-1 教師あり学習

### **線形回帰**

ここでは、教師あり学習のひとつとして線形回帰を考える。

回帰とは、手元にある実測値について**連続変数を用いたモデルで学習させること**を指す。線形回帰のモデルは$n$番目の予測値$y_n$に対して

$$
\begin{align}
y_n&=\sum_{m=1}^{M}w_mx_{nm}+b \\
&=\sum_{m=0}^{M}w_{m}x_{nm}
\end{align}
$$

と定義される。ここで、バイアス$b$は$w_0=b,x_0=1$として和の中に含めた。目標値ベクトル$\vec{y}$は行列$X=\{x_{nm}\}$と最適化パラメータベクトル$\vec{w}$を用いて

$$
\begin{align}
    \begin{pmatrix}
        y_1 \\ y_2 \\ \vdots \\ \\ \vdots \\ y_N
    \end{pmatrix}
    &=
    \begin{pmatrix}
        1 & x_{11} & \cdots & & \cdots & x_{1M} \\
        1 & x_{21} & \cdots & & \cdots & x_{2M} \\
        \vdots & \vdots & \ddots & &        & \vdots \\
                &        &        & &        &        \\
        \vdots & \vdots &        & & \ddots & \vdots \\
        1 & x_{N1} & \cdots & & \cdots & x_{NM}
    \end{pmatrix}
    \begin{pmatrix}
        w_0 \\ w_1 \\ \vdots \\ \\ \\ \\ \vdots \\ w_M
    \end{pmatrix} \\
    \therefore~~\vec{y}&=X\vec{w}
\end{align}
$$

と書ける。$\vec{w}$の$M+1$個全てに対して最適化を行う。以下では

1. 重回帰分析
2. Ridge回帰
3. Lasso回帰

### 重回帰分析

重回帰分析とは、2つ以上の変数から決定される値に対して回帰分析を行うことを指す。目的関数は**最小二乗法**による二乗和誤差であり、目標値を$t_n$と置けば

$$
\begin{align}
    \mathcal{L}&=\sum_{n=1}^{N}\left(t_n-y_n\right)^2 \\
    &=\left(\vec{t}-\vec{y}\right)^{\mathrm{t}}\left(\vec{t}-\vec{y}\right) \\
    &=\lVert\vec{t}-\vec{y}\rVert^2_2
\end{align}
$$

と書ける。最終式では$L2$ノルムを用いて表現した。式(4)を代入して$\vec{w}$について最適化の条件を求めると

$$
\begin{align}
    0&=\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\vec{w}}=0-2X^{\mathrm{T}}\vec{t}+\vec{w}^{\mathrm{T}}\left(X^{\mathrm{T}}X+\left(X^{\mathrm{T}}X\right)^{\mathrm{T}}\right) \\
    \therefore~~~\vec{w}&=\left(X^{\mathrm{T}}X\right)^{-1}X^{\mathrm{T}}\vec{t}
\end{align}
$$

式(9)を**正規方程式**という。式(9)を計算することで最適化パラメータを求めることができるが、実際にはデータセットが非常に多い場合には$\left(X^{\mathrm{T}}X\right)^{-1}$を計算するコストが大きくなるし、そもそも$X^{\mathrm{T}}X$に逆行列が存在することが非自明であるため、**勾配降下法**を用いて目的関数を最適化することのほうが多い(scikit-learnもこちらで計算しているはず)。

### Ridge回帰

Ridge回帰とは、重回帰モデルに対して**正則項**と呼ばれる項を導入し、$\vec{w}$の1成分だけが異常に大きい値になり、目的関数がその部分の入力変数の値に非常に敏感になってしまい、学習時に用いなかったデータに対して予測の精度が落ちてしまうこと(**過学習**)を防ぐことができる(ことがある)。

目的関数は次の通り。

$$
\begin{align}
    \mathcal{L}&=\sum_{n=1}^{N}\left(t_n-y_n\right)^2+\alpha\sum_{n=1}^{M}w_n^2 \\
    &=\lVert\vec{t}-\vec{y}\rVert^2_2+\alpha\lVert\vec{w}\rVert_2^2
\end{align}
$$

この目的関数の第二項が正則項と呼ばれる量で、係数$\alpha$はモデルの学習を行う前に設定するパラメータで、**ハイパーパラメータ**と呼ばれる。これによって目的関数に罰則を加えることで$\vec{w}$の値をバランスよくし、過学習を防ぐことができる。ちなみに、正規方程式は

$$
\begin{align}
    \vec{w}=\left(X^{\mathrm{T}}X+\alpha I\right)^{-1}X^{\mathrm{T}}\vec{t}
\end{align}
$$

と書け、正則項による変更が入っていることがわかる。こちらも勾配降下法を用いることが多いらしい。

### Lasso回帰

Lasso回帰とは、Ridge回帰と同様に重回帰モデルに対して正則項を導入して過学習を防ぐモデルである。目的関数は次の通り。

$$
\begin{align}
    \mathcal{L}&=\sum_{n=1}^{N}\left(t_n-y_n\right)^2+\alpha\sum_{n=1}^{M}\lvert w_n\rvert \\
    &=\lVert\vec{t}-\vec{y}\rVert^2_2+\alpha\lVert\vec{w}\rVert _1
\end{align}
$$

Ridge回帰との大きな違いは、正則項に$L1$ノルムが用いられている点である。これによってRidge回帰のように「最も大きいパラメータの値を小さくする」のではなく、「過学習の原因となっている入力変数のパラメータの値を0にしてしまう」という方法によって過学習を防ぐのである。正規方程式は難しいので[割愛](https://qiita.com/torahirod/items/a79e255171709c777c3a)...