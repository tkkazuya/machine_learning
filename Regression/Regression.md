# 教師あり学習 : 線形回帰

1. 線形回帰概要
2. 代表的な線形回帰モデルとその理論
    - 重回帰分析
    - Ridge回帰
    - Lasso回帰
    - Ridge回帰 vs Lasso回帰
3. 回帰分析の評価方法
    - 決定係数 $R^2$
    - k-Fold Cross Varidation
4. 相関関係と多重共線性
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

の3つを考える。

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

Ridge回帰とは、重回帰モデルに対して**正則項**と呼ばれる項を導入して目的関数を最適化する回帰モデルである。

重回帰分析の結果、$\vec{w}$ の成分の中で、相対的に大きい値を持つ成分が存在する場合を想定する。このような場合、目的関数がその重みの入力変数の値に非常に敏感になってしまい、予測の精度が落ちてしまうこと(**過学習**)がある。これに対し、Ridge回帰を用いることで大きな値の成分を小さくし、$\vec{w}$ のバランスを良くすることができる。これによって、重回帰分析よりも予測の制度を上げることができることがある。

重要なのは、**相対的に大きな値をもつ成分が存在することが、必ずしも予測の制度を下げるとは限らない**ことである。Ridge回帰によって $\vec{w}$ の分布のバランスを良くすることは確実に実行できるが、対象となったパラメータがその予測において最も大きな寄与を持つものだった場合、この値を小さくすることで精度を下げてしまうことが予想される。

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

Lasso回帰とは、Ridge回帰と同様に重回帰モデルに対して正則項を導入して過学習を防ぐモデルである。Ridge回帰と異なるのは、正則項が $L1$ ノルムであるという点にある。Lasso回帰の根底にあるのは、「**できるだけ少ない特徴量で現象を記述したい**」という考え方で、 $L1$ ノルムを導入することでこれが実現できる。

重回帰分析やRidge回帰の結果、特徴量が多く、その内重要なものがわずかしかないことが予想される場合を想定する。このような場合、重要でない特徴量が予測の精度に悪影響を及ぼしていると考えることができる。これに対し、Lasso回帰を用いると、不要な特徴量を排除することができ、予測精度を向上させることができるかもしれない。

Ridge回帰と同様に、重要なのは**特徴量が多いことが、必ずしも予測の制度を下げているとは限らないし、Lasso回帰によって消される特徴量がいつも重要でないものであるとは限らない**ことである。Lasso回帰によって最適化パラメータのいくつかを $0$ にしてしまうことは確実に実行できるが、そのパラメータ（特徴量）がその予測に大きな寄与を与えていた場合、予測の精度は落ちてしまうことが予想される。

目的関数は次の通り。

$$
\begin{align}
    \mathcal{L}&=\sum_{n=1}^{N}\left(t_n-y_n\right)^2+\alpha\sum_{n=1}^{M}\lvert w_n\rvert \\
    &=\lVert\vec{t}-\vec{y}\rVert^2_2+\alpha\lVert\vec{w}\rVert _1
\end{align}
$$

正規方程式は解析的には求められないっぽい。座標降下法と呼ばれる方法で実装されている。

参照：

- <http://taustation.com/lasso-regression-understanding/>
- <https://qiita.com/torahirod/items/a79e255171709c777c3a> 

### Ridge回帰 vs Lasso回帰

<table>
    <tr>
        <th></th>
        <th>メリット</th>
        <th>デメリット</th>
    </tr>
    <tr>
        <th>Ridge回帰</th>
        <td>・少ないデータ数でも機能する<br>・共線性があっても有用な変数が<br>削除されない</td>
        <td>・特徴量を絞ることができない</td>
    </tr>
    <tr>
        <th>Lasso回帰</th>
        <td>・解釈性、過学習防止性が高い</td>
        <td>・M>Nでは使えない<br>・共線性があるとき、特徴量を消してしまう<br>事がある</td>
    </tr>
</table>

## 3. 回帰分析の評価方法

以下では回帰分析に用いられる評価方法として

- 決定係数 $R^2$
- k-Fold Cross Varidation

を紹介する。

### 決定係数 $R^2$

決定係数 $R^2$ は次のような表式で表される。

$$
\begin{align}
    R^2=1-\frac{\sum_{k=1}^{n}\left(t_k-y_k\right)^2}{\sum_{k=1}^{n}\left(t_k-\bar{t}\right)^2} \notag
\end{align}
$$

$t_n$ は目標値、$\bar{t}$ は目標値の平均である。つまり、この決定係数は最小化した目的関数が目標値の分散に対してどれだけ離れているかを表している。したがって、そのモデルがデータセットに対してどれだけ当てはまりが良いかを見ることができる。

以前は計算機の性能の問題により、以下で述べるk-Fold Cross Varidationのような評価法を用いることができなかった。そのため、hold-out法でモデルを学習させ、その結果を決定係数で評価することが多かった。しかし、これは特徴量の数を増やすと値が良くなるという性質を持っており、モデルの評価を正しく行うことができないため、現代で使われることはない。また、それをある程度改善したものとして自由度調整済みの決定係数というものが存在するが、特徴量が増えるときに与える罰則の大きさが足りておらず、この問題を真に解決することはできない。

参照：

- <https://qiita.com/s-yonekura/items/43aefbe726ee814123f7#%E8%87%AA%E7%94%B1%E5%BA%A6%E8%AA%BF%E6%95%B4%E6%B8%88%E3%81%BF%E3%81%AE%E6%B1%BA%E5%AE%9A%E4%BF%82%E6%95%B0>
- <https://bellcurve.jp/statistics/course/9706.html>
- <https://marketing.xica.net/column/about-coefficient-of-determination/#:~:text=%E6%B1%BA%E5%AE%9A%E4%BF%82%E6%95%B0%E3%81%A8%E3%81%AF%E3%80%81%E5%9B%9E%E5%B8%B0,%E9%87%8D%E5%9B%9E%E5%B8%B0%E5%88%86%E6%9E%90%E3%81%8C%E3%81%82%E3%82%8A%E3%81%BE%E3%81%99%E3%80%82>

### k-Fold Cross Varidation

k-Fold Cross Varidationとは、以下のような手順で行うモデルの汎化性能を測る手段である。

1. 教師データを $a_1\sim a_k$ に $k$ 分割し、ある $a_i~(i=1,2,\cdots,k)$ をテストデータに、それ以外を学習データとしてモデルの学習を行い、精度 $r_i$ を算出する。
2. 1の操作を $i=1,2,\cdots,k$ で $k$ 回行い、出てきた $k$ 個の精度を平均したもの $\bar{r}$ を算出し、これをこのモデルの精度とする。

以下に $k=5$ の場合を図で示す。

![5-FoldCV](./5-FoldCV.png)
<https://datawokagaku.com/kfoldcv/#k-Fold_Cross_Validation>

この方法を用いることで、hold-out法で引き起こされるテストデータと学習データの分割にランダム性を減らすことができ( $k=N$ で完全に排除できる)、過学習を避けることにつながるため、モデルの精度評価をより正確に行うことができる。

ここで、精度として用いられるものは

- 平均平方二乗誤差（RMSE）
- 平均絶対誤差（MAE）

の二つがある。

#### 平均平方二乗誤差（RMSE）

これは以下のような式で与えられる。

$$
\begin{align}
    RMSE=\sqrt{\frac{1}{n}\sum_{k=1}^{n}\left(t_k-y_k\right)^2}
\end{align}
$$

予測値と目標値が近づくほどRMSEの値は小さくなる。誤差を二乗して足しているため、外れ値があると値が大きくなってしまう。したがって、RSMEは外れ値の影響を受けやすく、評価対象が正規分布に近いほど正確に評価できる。また、これは元のデータと同じ次元を持っているため、得られる結果は「見積もられる誤差の大きさ」として意味を持つ。

#### 平均絶対誤差（MAE）

これは以下のような式で表される。

$$
\begin{align}
    MAE=\frac{1}{n}\sum_{k=1}^{n}\left|t_k-y_k\right|
\end{align}
$$

これはRMSEと違い、誤差を二乗していないため外れ値に強い。得られる結果はRMSEと同様に「見積もられる誤差の大きさ」を表す。

最後に、この検証結果を元にしたモデルの最終選択について述べる。これは二つの主張がある。

- すべての学習データを用いて学習し直したモデルを最終結果とする
- k-Fold Cross Varidationで作った各モデルのパラメータを平均する

どちらがいいのかという部分は定量的な評価がなされているものを見つけることができなかったためよくわからないが、前者を指示している記事のほうが多かった（体感）気がする。

参照：

- <https://best-biostatistics.com/correlation_regression/crossvalidation.html#i-9>
- <https://aizine.ai/glossary-crossvalidation/>
- <https://qiita.com/oki_kosuke/items/3934cd311fc805cafe81>
- <https://www.simpletraveler.jp/2022/03/30/machinelearning-crossvalidation-model-selection/>
- <https://an-engineer-note.com/?p=17#toc5>

---

## 4. 相関関係と多重共線性問題

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
