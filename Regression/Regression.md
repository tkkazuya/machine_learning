# 教師あり学習 : 線形回帰

1. 線形回帰概要
2. 代表的な線形回帰モデルとその理論
    - 重回帰分析
    - Ridge回帰
    - Lasso回帰
3. 相関関係と多重共線性
    - PLS

---

## 1. 線形回帰概要

ここでは、教師あり学習のひとつとして線形回帰を考える。

回帰とは、手元にある実測値について**連続変数を用いたモデルで学習させること**を指す。線形回帰のモデルは $n$ 番目の予測値 $y_n$ に対して

$$
\begin{align}
y_n&=\sum_{m=1}^{M}w_mx_{nm}+b \\
&=\sum_{m=0}^{M}w_{m}x_{nm}
\end{align}
$$

と定義される。ここで、バイアス $b$ は $w_0=b,x_0=1$ として和の中に含めた。目標値ベクトル $\vec{y}$ は行列 $X=\{x_{nm}\}$ と最適化パラメータベクトル $\vec{w}$ を用いて

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

と書ける。$\vec{w}$ の $M+1$ 個全てに対して最適化を行う。以下では代表的な回帰モデルとして

1. 重回帰分析
2. Ridge回帰
3. Lasso回帰

の3つを考える。また、モデルの良し悪しを評価するための指標としては決定係数 $R$ というものが用いられることが多く、それは次のような表式で表される。

$$
\begin{align}
    R^2=1-\frac{\sum_{n=1}^{N}\left(t_n-y_n\right)^2}{\sum_{n=1}^{N}\left(t_n-\bar{t}\right)^2} \notag
\end{align}
$$

$t_n$ は目標値、$\bar{t}$ は目標値の平均である。つまり、この決定係数は最小化した目的関数が目標値の分散に対してどれだけ離れているかを表している。したがって、そのデータにとって良いモデルほど決定係数は1に近づき、そのモデルがどれだけ優れているかを数値で確認することができる。

---

## 2. 代表的な回帰モデル

### 重回帰分析

重回帰分析とは、2つ以上の変数から決定される値に対して回帰分析を行うことを指す。目的関数は**最小二乗法**による二乗和誤差であり、

$$
\begin{align}
    \mathcal{L}&=\sum_{n=1}^{N}\left(t_n-y_n\right)^2 \\
    &=\left(\vec{t}-\vec{y}\right)^{\mathrm{t}}\left(\vec{t}-\vec{y}\right) \\
    &=\lVert\vec{t}-\vec{y}\rVert^2_2
\end{align}
$$

と書ける。最終式では $L2$ ノルムを用いて表現した。式(4)を代入して $\vec{w}$ について最適化の条件を求めると

$$
\begin{align}
    0&=\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\vec{w}}=0-2X^{\mathrm{T}}\vec{t}+\vec{w}^{\mathrm{T}}\left(X^{\mathrm{T}}X+\left(X^{\mathrm{T}}X\right)^{\mathrm{T}}\right) \\
    \therefore~~~\vec{w}&=\left(X^{\mathrm{T}}X\right)^{-1}X^{\mathrm{T}}\vec{t}
\end{align}
$$

式(9)を**正規方程式**という。式(9)を計算することで最適化パラメータを求めることができるが、実際にはデータセットが非常に多い場合には $\left(X^{\mathrm{T}}X\right)^{-1}$ を計算するコストが大きくなるし、そもそも $X^{\mathrm{T}}X$ に逆行列が存在することが非自明であるため、**勾配降下法**を用いて目的関数を最適化することのほうが多い(scikit-learnもこちらで計算しているはず)。

### Ridge回帰

Ridge回帰とは、重回帰モデルに対して**正則項**と呼ばれる項を導入し、 $\vec{w}$ の1成分だけが異常に大きい値になり、目的関数がその部分の入力変数の値に非常に敏感になってしまい、学習時に用いなかったデータに対して予測の精度が落ちてしまうこと(**過学習**)を防ぐことができる(ことがある)。

目的関数は次の通り。

$$
\begin{align}
    \mathcal{L}&=\sum_{n=1}^{N}\left(t_n-y_n\right)^2+\alpha\sum_{n=1}^{M}w_n^2 \\
    &=\lVert\vec{t}-\vec{y}\rVert^2_2+\alpha\lVert\vec{w}\rVert_2^2
\end{align}
$$

この目的関数の第二項が正則項と呼ばれる量で、係数 $\alpha$ はモデルの学習を行う前に設定するパラメータで、**ハイパーパラメータ**と呼ばれる。この値の調整については、後の節で説明する。これによって目的関数に罰則を加えることで $\vec{w}$ の値をバランスよくし、過学習を防ぐことができる。ちなみに、正規方程式は

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

Ridge回帰との大きな違いは、正則項に $L1$ ノルムが用いられている点である。これによってRidge回帰のように「最も大きいパラメータの値を小さくする」のではなく、「過学習の原因となっている入力変数のパラメータの値を0にしてしまう」という方法によって過学習を防ぐのである。正規方程式は複雑なので[割愛](https://qiita.com/torahirod/items/a79e255171709c777c3a)...

---

## 3. 相関関係と多重共線性問題

多重共線性は、重回帰分析モデルにおいて入力変数の中に相関係数の高い組み合わせがあることを指す。これは、過学習を引き起こす原因の一つであり、回帰分析を行う上では避けては通れない問題である。この問題を解決するためには、相関が強い入力変数をデータセットから地道に取り除くアプローチが自然だが、面倒なのでモデルでなんとかしてみたい。そこで、次のPLSというアルゴリズムを考える。

### Partial Least Squares ( PLS )

これは、次のようなステップで実装されている。

1. 入力値と目標値の共分散が最大になるように主成分を抽出(取り出す成分の数はケースバイケース)
2. 抽出された主成分に対して重回帰分析を用いる。

これは、主成分分析と似た概念であるが、主成分分析のように「入力変数を線形変換した潜在変数の分散が最大になるように主成分(潜在変数)を抽出する」のではなく、教師あり学習ということを活かして「**目標値と入力値の共分散が最大になるように主成分を抽出する**」ことを行う。

潜在変数抽出について、簡単化した流れを以下の仮定の下で説明する。

- 元データは標準化しておき、それを表す行列は $X\in\mathbb{R}^{N\times D}$ とする。
- 抽出する潜在変数は $2$ 個として、それを表す行列を $Z\in\mathbb{R}^{N\times 2}$ とする。また、この行列を $2$ つの列ベクトルを用いて $Z\equiv(\vec{z}_{1}~\vec{z}_{2})$ のように書く。
- PLSのモデル式として以下を仮定する。 $$ \begin{align}
    X_1&=\vec{z}_{1}\vec{p}^{\mathrm{T}}_{1}+X_2,~~~~~~~~~\vec{f}_1=\vec{z}_{1}q_1+\vec{f}_2 \\
    X_2&=\vec{z}_{2}\vec{p}^{\mathrm{T}}_{2}+E,~~~~~~~~~\vec{f}_2=\vec{z}_{2}q_2+\vec{f}
\end{align} $$ ただし、$X_1=X,\vec{f}_1=\vec{y}$ であり、 $\vec{p}_{1},\vec{p}_{2}$ はローディング、 $E$ は残差と呼ばれる量である。

- 潜在変数 $Z$ は $X$ を線形変換 $$ \begin{align}
    \vec{z}_{n}=X_n\vec{w}_n~~~~~~~~~(n=1,2)
\end{align} $$ を行って得られるとする。ただし、$W\in\mathbb{R}^{D\times 2}, W\equiv(\vec{w}_{1}~\vec{w}_{2})$ である。また、ベクトル $||\vec{w}_1||_2=||\vec{w}_2||_2=1$ とする。

以上の仮定の下でPLSのモデル式を書き下すと次のようになる。


まず初めに、**$\vec{y}$ との共分散が最大になる $\vec{z}_1$** を求めることを考える。$||\vec{w}_1||_2=1$ という束縛条件の下で共分散を最大にすることを考えたいので、Lagrangeの未定乗数法を用いる。最大化する目的関数 $\mathcal{L}_1$ は

$$
\begin{align}
    \mathcal{L}_1=\vec{f}_1^{\mathrm{T}}\vec{z}_{1}-\lambda\left(||\vec{w}_1||_2^2-1\right)
\end{align}
$$

とかける。$\vec{z}_1=X_1\vec{w}_1$ と束縛条件を用いて

$$
\begin{align}
    \frac{\partial\mathcal{L}_1}{\partial\vec{w}_1}=0
\end{align}
$$

を計算すると、(詳細な計算は[ここ](https://datachemeng.com/partialleastsquares/)を参照)

$$
\begin{align}
    \vec{w}_1=\frac{X_1^{\mathrm{T}}\vec{f}_1}{||X_1^{\mathrm{T}}\vec{f}_1||_2}
\end{align}
$$

という結果を得ることができる。そして、$\vec{p}_1,q_1$ はそれぞれ $X_1,\vec{f}_1$ の要素の二乗和が最小(**最小二乗法**)になるように求める。結果は

$$
\begin{align}
    \vec{p}_1=\frac{X_1^{\mathrm{T}}\vec{z}_1}{\vec{z}_1^{\mathrm{T}}\vec{z}_1},~~~~~~~~~q_1=\frac{\vec{f}_1^{\mathrm{T}}\vec{z}_1}{\vec{z}_1^{\mathrm{T}}\vec{z}_1}
\end{align}
$$

と求められる。$\vec{z}_2$ についても同様に計算すると

$$
\begin{align}
    \vec{w}_2=\frac{X_2^{\mathrm{T}}\vec{f}_2}{||X_2^{\mathrm{T}}\vec{f}_2||_2},~~~~~~~~~\vec{p}_2=\frac{X_2^{\mathrm{T}}\vec{z}_2}{\vec{z}_2^{\mathrm{T}}\vec{z}_2},~~~~~~~~~q_2=\frac{\vec{f}_2^{\mathrm{T}}\vec{z}_2}{\vec{z}_2^{\mathrm{T}}\vec{z}_2}
\end{align}
$$

という結果になる(上の結果からほぼ自明)。$Z$ の成分が $3$ 以上の場合も次々計算していくことで求められる。

以上の計算によって、求めたかった最適化パラメータ $q_1,q_2$ を求めることができた。ここでの計算の本質は、**各パラメータ $q_1,q_2$ を求めるときに $X_1,X_2$ を用いていた**点である。$X_2$ は、$X$ から $\vec{z}_1$ の情報を引いたデータになっており、これをすることで $\vec{z}_1$ と $\vec{z}_2$ の相関がなくなる。これを**デフレーション**という。

PLSは、入力変数に相関があるデータ、サンプル数が入力変数よりも少ないデータ、ノイズが含まれているデータなどに用いると効果的であり、教師あり学習の強みを生かしたモデルである。

参照:
- <https://academ-aid.com/ml/pls#index_id7>
- <https://datachemeng.com/partialleastsquares/>
