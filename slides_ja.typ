#import "@preview/typslides:1.3.2": *
#import "@preview/ctheorems:1.1.3": *

// 定理環境の日本語化
#let theorem = thmbox(
  "theorem",
  "定理",
  fill: rgb("#e8f4fd"),
  stroke: rgb("#0074d9") + 1pt,
  padding: (top: 1em, bottom: 1em),
)

// 定義環境の日本語化
#let definition = thmbox(
  "definition",
  "定義",
  fill: rgb("#eefce8"),
  stroke: rgb("#2d9c02") + 1pt,
)

#show: thmrules

#show cite: set text(fill: black)

#show: typslides.with(
  ratio: "16-9",
  theme: rgb("1f6f78"),
  font: ("Noto Sans CJK JP", "DejaVu Sans"),
  font-size: 18pt,
  link-style: "underline",
  show-progress: true,
)

#front-slide(
  title: "Score-Based Diffusion Model",
  subtitle: [その導入と基礎],
  authors: "Yukinari Hisaki",
)

#table-of-contents()

#title-slide[
  なぜ拡散モデルなのか？
]

#slide(title: "Planningのための生成モデル")[
  - *The Goal:* 機械学習を用いて、複雑な人間の運転行動をどのようにモデル化すべきか？
  #v(0.5em)

  - *Simple Approach (Regression):*
    最も単純な方法は、コンテキスト $c$ を単一の未来の軌跡 $tau$ にマッピングする決定論的関数 $tau = f_theta (c)$ を学習することである。
    - 入力 $c$: 自車状態、他車、地図データなど。
    - 出力 $tau$: 状態系列 ${x_t, x_(t+1), ..., x_T}$。
  #v(0.5em)

  - *Limitations:*
    *しかし*、人間の行動は本質的に*マルチモーダル(多峰性)*であり、不確実である。
    - 曖昧な状況下では、取るべき行動が複数存在する。

  #v(0.5em)

  $arrow.double$ 単一の軌跡だけでなく、完全な分布 $p(tau | c)$ を学習する*生成モデル*が必要である。
]

#slide(title: "Motivation: Iterative Refinement")[
  #set text(size: 17pt)
  目標は、サンプル $x tilde q(x)$ から未知のデータ分布 $q(x)$ を推定することである。
  #v(0.8em)

  - *Standard Approach (e.g., VAE):*
    潜在変数 $z tilde cal(N)(0, I)$ を導入し、*単一*の Decoder $p_theta (x|z)$ を介してデータにマッピングする。
    $ z arrow.long^(p_theta (x|z)) x $

  #v(0.5em)

  - *Limitations:*
    *Single shot*の $p_theta (x|z)$ を介して複雑なデータ分布 $q(x)$ を学習することは困難である。
    単純な事前分布と複雑なデータ多様体の間のギャップが大きすぎるためである。
  #v(0.5em)

  - *Solution (Diffusion Models):*
    生成を*複数の段階的な変換*に分解する。
    サンプルを段階的に改善していく。
    $x_T arrow.long x_(T-1) arrow.long ... arrow.long x_0$
    $ x_T tilde cal(N)(0, I), x_0 tilde q(x_0) $
]

#slide(title: "3つの定式化と本日の焦点")[
  #set text(size: 16pt)

  「拡散モデル」のアイデアは、数学的に主に3つの方法で定式化できる:

  #v(0.5em)

  1. *Denoising Diffusion Probabilistic Models (DDPM) @ddpm @nonequilibrium-diffusion*
    - 離散マルコフ連鎖として定式化。
    - 変分下界 (ELBO) を介して最適化。
    - _キーアイデア:_ ノイズ付加をステップバイステップで逆転させることを学習する。
  #v(0.2em)

  2. *Score Matching @score-based-generative-models*
    - 確率微分方程式 (SDE) を介して定式化。
    - _キーアイデア:_ *スコア関数* $nabla_x log p_t (x)$(より高い密度への方向)を学習する。
  #v(0.2em)

  3. *Flow Matching @flow-matching*
    - 常微分方程式 (ODE) に基づく、より一般的なフレームワーク。
    - _キーアイデア:_ 確率パスを輸送する*ベクトル場* $v_t (x)$ を直接回帰する。
  #v(0.2em)
  *Today's Focus:*
  ここでは *2. Score Matching* に焦点を当てる。
]

#title-slide[
  スコアマッチングの基礎
]

#slide(title: "動機: 非正規化分布の学習")[

  #set text(size: 14pt)

  *目標:*
  未知の分布 $q (x)$ からのサンプルデータセット ${x_1, ..., x_N}$ が与えられたとき、それを近似するモデル $p_theta (x)$ を学習したい。
  #v(0.5em)

  - *モデリングの課題:*
    柔軟な確率モデルは、しばしば正規化定数(分配関数)を除いた形で定義される(Energy-Based Model; EBM)。
    $ p_theta (x) = exp(-E_theta (x)) / Z_theta, quad Z_theta = integral exp(-E_theta (x)) dif x $

  #v(0.5em)

  - *計算不能な分配関数:*
    最尤推定 (MLE) で学習するには、$log p_theta (x)$ を最大化する必要がある。
    しかし、$Z_theta$(高次元空間での積分)の計算は計算量的に不可能である。
    $ log p_theta (x) = -E_theta (x) - underbrace(log Z_theta, "計算不能！") $

  #v(0.5em)

  - *解決策(微分):*
    入力 $x$ に関して勾配をとると、定数 $Z_theta$ は消える！
    $ nabla_x log p_theta (x) = - nabla_x E_theta (x) - underbrace(nabla_x log Z_theta, "0") = - nabla_x E_theta (x) $
]

#slide(title: [スコア関数: 定義と直感])[

  #set text(size: 16pt)

  密度の代わりに「対数密度の勾配」をモデル化することで、分配関数の問題を完全に回避できる。
  #definition("スコア関数")[
    確率密度 $p(x)$ に対して、スコアは以下のように定義される:
    $ s(x) := nabla_x log p(x) $
  ]

  - *直感: ベクトル場*
    - スコアは、*最急上昇*(より高いデータ密度)の方向を指すベクトル場である。
    - 重要なのは、$Z_theta$ を計算することなく有効な学習ターゲットを作成できることだ。
  #v(0.5em)

  - *目的関数(スコアマッチング):*
    ニューラルネットワーク $s_theta (x)$ をデータのスコアに一致させるように学習する:
    $
      cal(L)_("SM")(theta) = bb(E)_(x tilde q ) [ ||
        s_theta (x) - nabla_x log q (x) ||^2_2 ]
    $
]

#slide(title: [デノイジング・スコアマッチング: 計算可能なターゲット @score-matching])[

  #set text(size: 18pt)
  *問題:* 真のスコア $nabla_x log q(x)$ は計算不能である。
  #set text(size: 16pt)
  #theorem("明示的スコアマッチングとデノイジング・スコアマッチングの等価性")[
    $q(x)$ をデータ分布、$q(tilde(x)|x)$ を摂動カーネルとする。
    平滑化された周辺分布を $q(tilde(x)) = integral q(tilde(x)|x)q(x) dif x$ と定義する。
    平滑化周辺分布に対する明示的スコアマッチング目的関数
    $
      cal(L)_("ESM")(theta) = bb(E)_(q(tilde(x))) [ ||
        s_theta (tilde(x)) - nabla_tilde(x) log q(tilde(x)) ||_2^2 ],
    $
    を最適化することは、$theta$ に依存しない定数 $C$ を除いて、以下のデノイジング・スコアマッチング目的関数を最適化することと等価である:
    $
      cal(L)_("DSM")(theta) = bb(E)_(q(x)q(tilde(x)|x)) [ ||
        s_theta (tilde(x)) - nabla_tilde(x) log q(tilde(x)|x) ||^2_2 ].
    $
    $ => cal(L)_("ESM")(theta) = cal(L)_("DSM")(theta) + C $
  ] <denoising-score-matching-theorem>

  #pagebreak()

  #set text(size: 18pt)

  *Core Intuition:*
  周辺スコア $nabla_tilde(x) log q(tilde(x))$ の計算は、$q(tilde(x)) = integral q(tilde(x)|x)q(x) dif x$ が未知のデータ分布 $q(x)$ に依存するため計算不能である。
  この定理により、$q(tilde(x)|x)$ に依存する $cal(L)_("DSM")$ を代わりに使用できるようになる。
  *重要な点として*、摂動カーネル $q(tilde(x)|x)$ は未知の確率分布 $q(x)$ から独立している。
  このカーネルは我々自身が設計するため、そのスコア $nabla_tilde(x) log q(tilde(x)|x)$ は解析的に計算可能である。
  *条件付き分布の選択:*
  典型的には、$q(tilde(x)|x)$ として*分散の小さいガウス分布*を選択する(例: $cal(N)(tilde(x); x, sigma^2 I)$)。
  これにより、条件付きスコアは解析的に扱いやすくなり、
  $ nabla_tilde(x) log q(tilde(x)|x) = - (tilde(x) - x) / sigma^2 $
  $q(tilde(x))$ はデータ分布 $q(x)$ を密接に近似できるようになる。
]

#slide(title: [Sampling: スコアを用いたLangevin Dynamics @ncsn])[

  スコア関数 $s_theta (x) approx nabla_x log q(x)$ を手に入れた後、どのようにサンプルを生成するか？
  #v(0.5em)

  - *Langevin Dynamics:*
    *$x_0$ を任意に初期化*し、スコアを使用して反復的に更新する:
    $
      x_(t+1) arrow.l x_t + underbrace(epsilon_t s_theta (x_t), "勾配上昇") + underbrace(sqrt(2 epsilon_t) z_t, "ランダムノイズ")
    $
    ここで、$z_t tilde cal(N)(0, I)$ であり、$epsilon$ はステップサイズである。
  #v(0.5em)

  - *メカニズム:*
    1. *勾配項:* $x$ を高密度領域(モード)へ押し上げる。
    2. *ノイズ項:* 単一のモードへの崩壊を防ぎ、分布全体の探索を可能にする。
  #v(0.5em)

  $arrow.double$ $t arrow infinity$ かつ $epsilon_t arrow 0$ とすると、$x_t$ は $q(x)$ からのサンプルに収束する。
]

#slide(title: "なぜ単純なスコアマッチングは失敗するのか")[

  #set text(size: 15pt)

  理論自体は健全だが、これをナイーブに適用した学習は実際にはしばしば失敗する。
  #v(0.5em)

  1. *多様体仮説 (The Manifold Hypothesis)*
    - 高次元データは低次元の多様体上に集中している。
    - この多様体の外側では、$q(x) approx 0$ である。
    - スコア $nabla_x log q(x)$ は*未定義*か、数値的に不安定である。
  #v(0.5em)

  2. *低密度領域での不正確な推定*
    - 損失 $bb(E)_(x tilde q)$ は、*データが存在する場所*でのみ誤差を最小化する。
    - 低密度領域(空間の大部分)では、モデル $s_theta (x)$ は学習されない。
    - *結果:* ランダムノイズ(低密度)からサンプリングを開始すると、勾配が不正確になり、ゴミのような生成結果になる。
  #v(0.5em)

  3. *Langevin Dynamicsの遅い混合 (Slow Mixing)*
    - データ分布が互いに素なモード(孤立したデータの「島」)を持つ場合、その間のスコアはほぼゼロになる。
    - サンプラーは1つのモードにスタックし、他のモードへのギャップを越えられなくなる。
]

#title-slide[
  拡散の導入
]

#slide(title: "ノイズとデータの橋渡し")[

  #set text(size: 16pt)

  確率が定義されていない領域でスコアを計算することを避けるにはどうすればよいか？
  #v(0.5em)

  - *Strategy: 連続的な分布の変形*
    $q$ を直接モデル化する代わりに、以下をつなぐ $t in [0, T]$ に対する分布の変形 $p_t$ を考える:
    - $p_0 approx q$ (複雑なデータ分布)。
    - $p_T approx cal(N)(0, I)$ (どこでも定義される扱いやすいノイズ分布)。
  #v(0.5em)

  - *2つの重要な課題:*
    1. *学習:* $p_t$ 自体が未知の場合、どのようにして時間依存スコア $s_t (x) = nabla_x log p_t (x)$ を学習するか？
    2. *サンプリング:* スコアが未定義の領域に足を踏み入れずに、どのようにサンプルを生成するか？
  #align(center)[
    #image("assets/diffusion_sde_only.png", width: 50%)
  ]
]

#slide(title: "課題1: Forward SDEの設計と学習")[

  #set text(size: 16pt)

  データ分布 $p_0$ は未知であるため、中間分布 $p_t$ を数学的に直接定義することはできない。
  代わりに、各データ点を時間発展させる*プロセス*を定義する。

  #v(0.5em)

  - *The Forward SDE:*
    データ点を徐々にノイズに変換する拡散過程を構築する:
    $
      dif x = underbrace(f(x, t), "ドリフト:決定論的変化") dif t
      + underbrace(g(t), "拡散:ランダムノイズ") dif w,
      quad x_0 tilde p_0(x_0)
    $

  #v(0.5em)

  - *Forward SDE の設計:*
    以下の2つの重要な条件を満たすように SDE を*設計*する必要がある:

    1. *計算可能な遷移:*
      遷移確率 $p_(t|0) (x_t | x_0)$ は扱いやすいガウス分布でなければならない。これにより、学習中に SDE を step-by-step でシミュレーションする必要がなくなる。
    2. *単純な事前分布:*
      プロセスは標準ガウス分布 $p_T (x) approx cal(N)(0, I)$ に収束し、サンプリングの既知の開始点を提供しなければならない。
  #v(0.5em)

  $arrow.double$ 条件 1 により $p_(t|0) (x_t | x_0)$ が得られるため、未知の $p_t (x)$ を*デノイジング・スコアマッチング*を使用してスコアネットワーク $s_theta (x, t)$ を学習できる。
]

#slide(title: "課題2: 確率フローによるサンプリング")[

  #set text(size: 14pt)

  すべてのタイムステップのスコア $s_theta (x, t)$ が学習された後、どのようにサンプリングするか？
  単純な Langevin dynamics では、スコアが不正確な低密度領域に踏み込んでしまう可能性がある。
  #v(0.5em)

  - *確率フロー ODE (Probability Flow ODE):*
    すべての SDE に対して、*その周辺分布 $p_t (x)$ がすべての $t$ において SDE のものと等しい*決定論的常微分方程式(ODE)が存在する(後ほど詳しく説明)。
  #v(0.5em)

  - *サンプリングのメカニズム:*
    1. サンプル $x_T tilde p_T$ (単純なガウス分布)。
    2. 時間を $T$ から $0$ へ逆方向に ODE を解く。
  #v(0.5em)

  $arrow.double$ 明確に定義された領域から始まる有効な確率フローに従うため、軌跡 $x_t$ は常にスコアが定義される領域に留まり、未定義のスコア領域を回避する。
  #align(center)[
    #image("assets/diffusion.png", width: 50%)
  ]

]

#slide(title: "ロードマップ: モデルの構築")[

  #set text(size: 16pt)

  ここからは、3つの具体的なステップで実際のモデルを構築する:

  #v(0.5em)

  1. *ステップ 1: SDEの設計 (VP SDE)*
    ガウスノイズへの収束を保証するために、特定のドリフト $f(x, t)$ と拡散 $g(t)$ を定義する。
  2. *ステップ 2: スコアの学習*
    勾配場 $nabla_x log p_t(x)$ を推定するためにニューラルネットワーク $s_theta (x, t)$ を学習する。
  3. *ステップ 3: 生成 (Reverse ODE)*
    学習したスコアを確率フロー ODEに代入し、ノイズからデータへ逆方向に解く。
]

#title-slide[
  ステップ 1: SDEの具体的な設計 (VP SDE)
]

#slide(title: "VP SDE: 標準的な確率フロー")[

  #set text(size: 16pt)

  確率フローの標準的な選択肢は *Variance Preserving (VP) SDE* であり、これは DDPMの連続時間極限である。
  #v(0.5em)

  - *SDEの定式化:*
    時間依存ノイズスケジュール $beta(t) > 0$ を用いてドリフトと拡散を定義する:
    $
      dif x = underbrace(-1/2 beta(t) x, "ドリフト: 信号を縮小") dif t + underbrace(sqrt(beta(t)), "拡散: ノイズを注入") dif w
    $

  #v(0.5em)

  - *なぜ「分散保存 (Variance Preserving)」なのか？*
    初期データ $x_0$ が分散 $1$を持つ場合、$x_t$ の分散はすべての $t$ に対して $1$ のままである。
    ドリフトはデータを原点に向かって滑らかに縮小し、拡散はガウスノイズを追加することでそれを完全に補償する。
  - *計算可能な遷移カーネル:*
    この SDE は線形であるため、$x_0$ から $x_t$ への遷移確率は常に扱いやすいガウス分布となる。
    $alpha_t = exp(- integral_0^t beta(s) dif s)$ とすると:
    $
      p_(t|0)(x_t | x_0) = cal(N)(x_t; sqrt(alpha_t) x_0, (1 - alpha_t) I)
    $

  #pagebreak()


  - *実際の実装(線形スケジュール):*
    実際には、$t in [0, 1]$ とし、$beta(t)$ を線形関数として定義する:
    $ beta(t) = beta_"min" + t (beta_"max" - beta_"min") $
    標準的なハイパーパラメータは $beta_"min" = 0.1$ および $beta_"max" = 20$ である。
    $ integral_0^t beta(s) dif s = beta_"min" t + 1/2 (beta_"max" - beta_"min") t^2 $
    $ => alpha_t = exp(- beta_"min" t - 1/2 (beta_"max" - beta_"min") t^2) $

  #v(0.5em)

  $arrow.double$ $t=1$ において、$alpha_1 approx exp(-10) approx 4.5 times 10^(-5) approx 0$ となる。
  信号は完全に破壊され、$p_1(x) approx cal(N)(0, I)$ が保証される。

  #pagebreak()

  - *時間発展の可視化 ($x_0 = 1$):*
    これらの方程式はデータをどのように変換するか？
    $x_0 = 1$ から始まる単一の1次元データポイントを時間 $t in [0, 1]$ にわたって追跡する。
  #v(0.5em)

  #align(center)[
    #image("assets/vp_sde_plot.png", width: 50%)
  ]

  #v(0.5em)

  - *ドリフト (平均):*
    平均 $mu(t) = sqrt(alpha_t) x_0$ は $1$ から $0$ へ滑らかに減衰する。
    元の信号は体系的に消去される。
  - *拡散 (分散):*
    $plus.minus sigma(t)$ の区間(ここで $sigma(t) = sqrt(1 - alpha_t)$)は、$t=1$ で分布が標準ガウスノイズ $cal(N)(0, 1)$ に完全に一致するまで着実に拡大する。
]

#title-slide[
  ステップ 2: スコアの学習
]

#slide(title: "学習: 連続時間スコアマッチング")[

  #set text(size: 16pt)

  具体的な SDE とその遷移カーネル $p_(t|0)(x_t | x_0)$ を設計できた。
  次は、すべての連続時間 $t in [0, 1]$ にわたってスコア $nabla_x log p_t (x)$ を近似するようにニューラルネットワーク $s_theta (x, t)$ を学習する。
  #v(0.5em)

  - *デノイジング・スコアマッチングの拡張:*
    先ほどと同様に、周辺スコア $nabla_x log p_t (x)$ は直接計算できない。
    しかし、周辺スコアを直接マッチングする代わりに、*既知の*条件付きスコア $nabla_(x_t) log p_(t|0)(x_t | x_0)$ をマッチングさせることと数学的に等価である。
  #v(0.5em)

  - *VP SDEの条件付きスコア:*
    遷移カーネルは $p_(t|0)(x_t | x_0) = cal(N)(x_t; sqrt(alpha_t) x_0, (1 - alpha_t) I)$ であるため、スコアは単純にガウス対数密度の勾配となる:
    $
      nabla_(x_t) log p_(t|0)(x_t | x_0) = - (x_t - sqrt(alpha_t) x_0) / (1 - alpha_t)
    $
  #pagebreak()

  #set text(size: 15pt)

  - *連続時間の目的関数:*
    この条件付きスコアへの期待距離を最小化することで、スコアネットワーク $s_theta (x_t, t)$ を学習する。
    時間 $t$ を一様に、元データ $x_0$、および摂動データ $x_t$ をサンプリングする。
    $
      cal(L)(theta) = bb(E)_(t tilde cal(U)(0, 1)) bb(E)_(x_0 tilde p_0) bb(E)_(x_t tilde p_(t|0)) [ lambda(t) ||
        s_theta(x_t, t) - underbrace((- (x_t - sqrt(alpha_t) x_0) / (1 - alpha_t)), nabla_(x_t) log p_(t|0)(x_t | x_0)) ||^2_2 ]
    $

  #v(0.5em)

  - *重み付け関数 $lambda(t)$:*
    スコアの大きさは $t arrow 0$ につれて極めて大きくなる(分散 $(1 - alpha_t) arrow 0$ となるため)。
    勾配爆発を防ぎ、すべての時間で損失のバランスをとるために、典型的には分散に比例する正の重み付け関数、例えば $lambda(t) = 1 - alpha_t$ を選択する。
  #v(0.5em)

  $arrow.double$ この単一の目的関数を最適化することで、$s_theta (x, t)$ は連続ベクトル場を学習できる。
]

#title-slide[
  ステップ 3: 生成 (Reverse ODE)
]

#slide(title: "理論: 確率フロー ODE")[

  #set text(size: 14pt)

  データを生成するには、拡散過程を逆転させる必要がある。しかし、SDE の符号を反転させるだけでは逆過程を得ることはできない。
  そこで、確率過程(SDE)と決定論的過程(ODE)を結びつける基本定理を利用する。
  #theorem([確率フロー ODE @score-based-generative-models])[
    任意の SDE $dif x = f(x, t) dif t + g(t) dif w$ に対して、対応する決定論的な*常微分方程式 (ODE)* が存在する。
    この ODE は、すべての時間 $t$ において SDE と*全く同じ周辺確率密度* $p_t (x)$ を共有する:
    $
      dif x = [ f(x, t) - 1/2 g(t)^2 nabla_x log p_t (x) ] dif t
    $
  ] <probability-flow-ode-theorem>


  - *含意:*
    もしスコア $nabla_x log p_t (x)$ がわかれば、ノイズ分布 $p_T$ からデータ分布 $p_0$ へ決定論的にサンプルを輸送できる。

    #align(center)[
      #image("assets/diffusion.png", width: 50%)
    ]
]

#slide(title: "アルゴリズム: Reverse ODEを解く")[

  #set text(size: 16pt)

  実際には、未知の真のスコアを学習済みニューラルネットワーク $s_theta (x, t)$ に置き換える。
  #v(0.5em)

  - *Generation:*
    VP SDE の項($f = -1/2 beta(t)x$, $g = sqrt(beta(t))$)を代入し、以下の方程式を $t=T$ から $0$ へ*逆方向*に解く:
    $
      dif x = [ -1/2 beta(t) x - 1/2 beta(t) s_theta (x, t) ] dif t
    $

  #v(0.5em)

  - *サンプリングステップ:*
    1. *初期化:* 純粋なノイズ $x_T tilde cal(N)(0, I)$ をサンプリングする。
    2. *解く:* 数値解法(例:Euler、Runge–Kutta、または DPM-Solver)を使用して、$T$ から $0$ まで ODE を積分する。
    3. *出力:* 最終状態 $x_0$ が生成されたデータとなる。
]

#title-slide[
  まとめ: 拡散フレームワーク
]

#slide(title: "まとめ: 拡散フレームワーク")[

  確率フロー ODE に基づく生成モデルを構築した。
  #v(0.5em)

  1. *順方向プロセス (VP SDE):*
    データを滑らかにガウスノイズに劣化させる線形 SDE を設計し、扱いやすい遷移カーネル $p_(t|0) (x_t | x_0)$ を提供した。
  2. *学習 (スコアマッチング):*
    解析的な条件付きスコアとマッチングさせることで、計算不能な周辺密度の正規化定数を回避してスコア関数 $s_theta (x, t)$ を学習した。
  3. *生成 (Reverse ODE):*
    学習したスコアを使用して、正規分布 $x_T tilde p_T$ からデータ分布 $x_0 tilde p_0 approx q$ へサンプルを輸送する決定論的 ODE を逆方向に解くことで、新しいデータの生成を可能にした。
]

#title-slide[
  制御可能な生成: SDEのGuidance
]

#slide(title: "Conditional Generation: Bayes' Rule")[

  #set text(size: 16pt)

  これまでの説明で無条件のデータ $x tilde p_0(x)$ を生成する方法は明らかになった。
  もしクラス $y$(例:テキストプロンプトやクラスラベル)を条件としてデータを生成したい場合はどうするか？
  #v(0.5em)

  - *条件付きスコア:*
    条件付きサンプルを生成するには、Reverse ODE 内の無条件スコア $nabla_x log p_t (x)$ を条件付きスコア $nabla_x log p_t (x | y)$ に置き換えるだけでよい。
  #v(0.5em)

  - *ベイズの定理の適用:*
    事後確率 $p_t (x | y) prop p_t (x) p_t (y | x)$ の勾配をとることで、以下のような分解が得られる:

    $
      nabla_x log p_t (x | y) = underbrace(nabla_x log p_t (x), "unconditional score" \ s_theta (x, t)) + underbrace(nabla_x log p_t (y | x), "Guidance 項" \ ("尤度"))
    $

  #v(0.5em)

  - *メカニズム:*
    1. *無条件スコア:* 生成されたサンプルが(データから学習した)リアルで高品質なデータに見えることを保証する。
    2. *Guidance 項:* ベクトル場として機能し、軌跡を条件 $y$ を満たすデータ多様体上の所望の領域へと押し込む。
]

#slide(title: "応用: Diffusion PlannerにおけるGuidance")[

  #set text(size: 16pt)

  *Diffusion Planner* では、特定の制約(例:障害物回避、目標速度)を満たす有効な軌跡を生成したい。
  #v(0.5em)

  - *Rethinking the Guidance Term:*
    $p_t (y | x)$ を予測するために別のニューラルネットワーク分類器を学習する代わりに、これらのルールを*人間が設計したコスト関数* $cal(E)(x)$ を用いて数学的に定義できる。

    $arrow.double$ 制約を満たす軌跡の確率を高く、満たさない軌跡の確率を低くするようなスコアを定義できる。

  #v(0.5em)

  - *Energy-Based Formulation:*
    条件 $y$ をボルツマン分布として定式化する。
    軌跡が有効である確率は、コストが増加するにつれて指数関数的に減少する:
    $ p_t (y | x_t) prop exp(- lambda cal(E)(x_t)) $ <energy-based-formulation>
    ここで、$lambda > 0$ は Guidance の強さを制御し、$cal(E)(x_t)$ は総コストである。
  #pagebreak()

  - *Relationship between Trajectory Optimization and Guidance:*
    $cal(E)(x_t)$ は一般的な軌道最適化のコスト関数として捉えることができる。Guidance を適用した生成は、学習による生成と軌道最適化の双方を組み合わせたものと解釈できる。


  - *Analytic Guidance Field:*
    $ nabla_(x_t) log p_t (x_t | y) = s_theta (x_t, t) - lambda nabla_(x_t) cal(E)(x_t) $
    *(例:$cal(E)(x_t) = cal(E)_"obstacle" (x_t) + cal(E)_"velocity" (x_t)$)*

  #v(0.5em)

  $arrow.double$ 無条件スコア $s_theta (x_t, t)$ は人間の運転行動を模倣し、一方で $-lambda nabla_(x_t) cal(E)(x_t)$ は推論時に軌跡を設計された制約に沿う方向へ誘導する「外力」として機能する。
]

#title-slide[
  ODEが可能にする高度な機能
]

#slide(title: "応用1: 潜在空間補間")[

  #set text(size: 16pt)

  確率フロー ODE は完全に*決定論的*であるため、複雑なデータ空間 $x_0 tilde p_0$ と単純な潜在空間 $x_T tilde p_T$ の間に全単射(1対1)写像を定義する。
  #v(0.5em)

  - *補間の方法: 画像の補間の例*
    1. *Encode:* 2つの実画像 $x_0^((A))$ と $x_0^((B))$ を用意する。
    ODE を*順方向*にシミュレーションし ($t=0 arrow T$)、それらの正確な潜在表現 $x_T^((A))$ と $x_T^((B))$ を見つける。
    2. *補間:* 潜在空間でパラメータ $lambda in [0, 1]$ を使用して潜在変数を作成する(例:$z_lambda = "Slerp"(x_T^((A)), x_T^((B)), lambda)$)。
    3. *デコード:* $z_lambda$ から開始して ODE を*逆方向*にシミュレーションし ($t=T arrow 0$)、中間画像 $x_0^((lambda))$ を生成する。
  #v(0.5em)

  - *応用方法:*
    たとえば、経路生成の文脈で潜在空間を活用すると、異なるプランナーから出力される経路を潜在空間上で滑らかに補間し、経路の急な変化を防ぐというような応用が考えられる。
  #align(center)[
    #image("assets/celeba_interp.png", width: 65%)
  ]
]

#slide(title: "応用2: 正確な対数尤度の計算")[

  #set text(size: 16pt)

  Reverse ODE により、任意のデータ点の*正確な*確率密度 $p_0(x)$ を計算できる。
  #v(0.5em)

  - *瞬時変数変換公式:*
    $f_("ODE")(x, t)$ を確率フロー ODE のベクトル場とする。
    対数確率の時間変化は、このベクトル場の負の発散(ダイバージェンス)によって与えられる:
    $
      (partial log p_t (x)) / (partial t) = - nabla_x dot.c f_("ODE")(x, t)
    $

  #v(0.5em)

  - *尤度の計算:*
    実画像 $x_0$ の正確な対数尤度を求めるには、$t=0$ から $t=T$ まで ODE 軌跡に沿ってこの発散を積分する:
    $
      log p_0(x_0) = log p_T (x_T) + integral_0^T nabla_(x_t) dot.c f_("ODE")(x_t, t) dif t
    $
    ここで、$p_T (x_T)$ は単純なガウス事前分布 $cal(N)(0, I)$ である。
  #v(0.5em)

  - *応用方法:* 尤度を用いることによって、モデルの評価や異常検知が可能になる。
]

#title-slide[
  生成の高速化: DPM-Solver++
]

#slide(title: "生成の高速化: DPM-Solver++")[

  確率フロー ODE は、標準的な数値解法(Euler や Runge–Kutta など)で解くのは非常に遅く、多くの場合、200〜1000ステップを必要とする。
  #v(0.5em)

  - *なぜそんなに遅いのか？*
    ODE には「剛性(stiff)」のある線形ドリフト項が含まれている。
    大きなステップサイズをとると、打ち切り誤差が爆発し、画像が破壊される。
    $
      dif x = underbrace(-1/2 beta(t) x, "線形ドリフト (剛性あり)") dif t - 1/2 beta(t) underbrace(epsilon_theta (x, t), "ニューラルネットワーク") dif t
    $

  #pagebreak()

  - *半解析的解法(指数積分器):*
    ODE 全体をブラックボックスとして扱うのではなく、その半線形構造を利用する。
    定数変化法(variation of constants formula)を適用することで、線形部分 $-1/2 beta(t) x$ を*正確に解析的に*解くことができる。
    $
      x(t) = underbrace(e^(- integral_s^t 1/2 beta(r) dif r) x(s), "正確な線形発展") - integral_s^t e^(- integral_u^t 1/2 beta(r) dif r) 1/2 beta(u) epsilon_theta (x_u, u) dif u
    $

    #v(0.5em)

    $arrow.double$ ニューラルネットワーク部分 $epsilon_theta$ の積分のみを数値的に近似すればよい。
    $epsilon_theta$(またはデータ予測 $x_theta$)は生のドリフトよりもはるかに滑らかに変化するため、最小限の誤差で巨大なステップをとることができる！
]

// Bibliography
#let bib = bibliography("bibliography.bib")
#bibliography-slide(bib)
