# Evaluating Cross-Video Emotion Recognition in EEG with Comparative Study and Factor Exploration

[JSAI 2024](https://img.shields.io/badge/JSAI-2024-blue)

[IEEE PICom 2024](https://img.shields.io/badge/IEEE-PICom%202024-00629B)

# 概要

---

生理信号（例：EEG）に基づく情動識別研究では、モデルが特定の動画刺激の特徴を「記憶」してしまい、その結果、クロスビデオ汎化（すなわち学習に用いた動画と評価に用いる動画が異なる設定）において性能が著しく低下することが少なくありません。本研究は Cross-Video Emotion Recognition の課題設定に焦点を当て、データ前処理から学習・評価に至るまでの一連の実験パイプラインを体系的に整理し、実験の再現、アブレーション、ならびにモデル拡張を容易にすることを目的としています。

## 研究目的

本研究は、Cross-Video Emotion Recognition（クロスビデオ情動識別） ****の課題に着目し、EEG 信号のみに基づく深層学習手法の汎化性能を体系的に検討することを目的です。具体的な研究目的は以下のとおりです。

- **EEGデータのみを用いた現代のディープラーニングアーキテクチャにおける、異なる刺激源間での汎化性能の検証。**
- **クロスビデオ情動識別における精度に影響を及ぼす要因の特定と分析。**
- **高い汎化性能を備えた感情認識モデルの構築に向けた、実践的な知見および指针の提供。**

## 研究結果

### 非クロスビデオ情動識別

非クロスビデオ（動画内）実験では、DEAPデータセットを用い、2D-CNNをベンチマークとして5分割交差検証を行いました。その結果、覚醒度（Arousal）で**95.92%**、価数（Valence）で**96.24**%という高い平均精度を達成しました。全動画のデータを使用した場合の精度は極めて高い一方、**未知の動画（Unseen videos）**に対する汎化性能の向上が本研究の主要な課題となっています。

![image.png](image\image.png)

### クロスビデオ情動識別

クロスビデオ実験では、**動画1件抜き交差検証 (LOVO)** を用いてモデルの汎化性能を評価しました。実験の結果、**特徴量ベースの2D-CNN** が最も高い覚醒度（Arousal）精度を達成し（多くが50%超）、LOVOにおいて最も高いポテンシャルを示しました。一方、エンドツーエンドのCNN-Transformerはアンダーフィッティングに陥り、ファインチューニング手法もデータ削減時には事前学習の優位性が失われることが確認されました。動画内実験での高精度（約96%）と比較して、**未知の動画 (Unseen videos)** に対する認識精度は大幅に低下しており、汎化性能の向上が極めて重要な課題であることが浮き彫りとなりました。

![image.png](image\image1.png)

### 各脳波帯域がクロスビデオ情動識別に及ぼす影響

![image.png](image\image2.png)

### 感情ラベルの正確性がクロスビデオ情動識別に及ぼす影響

![image.png](image\image3.png)

![image.png](image\image4.png)

![image.png](image\image5.png)

# **はじめに**

## 前提条件（Prerequisites）

✅ **Python 実行環境**（Conda または venv の使用を推奨します）

✅ **（任意）NVIDIA GPU + CUDA**（学習の高速化を行う場合）

✅ **データセットの原始ファイルを取得済みであること**（本リポジトリにはデータは含まれていません）

DEAP / AMIGOS / DREAMER の入手方法については、各データセットの公式サイトおよびライセンス規約に従ってください。

## インストール手順（Installation）

```
git clone https://github.com/miracle-huang/cross-video-emotion-recognition.git
cd cross-video-emotion-recognition

pip install -r requirements.txt
```

## 実行方法（Usage）

本リポジトリでは、**データ前処理**と**異なるモデルの学習**をそれぞれ独立したスクリプトとして実行します。各処理の詳細な設定は、対応するスクリプトおよび設定ファイルをご確認ください。

```
# 2D-CNN
python run/run_2d_cnn_leave_one_video_out.py

# CNN-Transformer
python run/run_cnn_transformer.py
```

<aside>
⚠️

注意

</aside>

- 実行に必要なスクリプト名、引数、および設定ファイルは、実験設定によって異なります。
- 本 README では、具体的な実行コマンドについては**今後の整理・更新が必要**です。

## 実験設定 （Experimental Settings）

本研究では、**EEG に基づく Cross-Video Emotion Recognition（CV-ER）** におけるモデルの汎化性能を評価するため、複数のデータセットおよび深層学習モデルを用いた比較実験と要因分析を実施します。

### 情動表現

情動は **Russell の円環モデル** に基づき、以下の 2 次元で表現します。

- **Valence**：情動の快・不快の度合い
- **Arousal**：情動の覚醒レベル

いずれも **High / Low の二値分類**として設定し、被験者の主観評価スコアに基づいてラベルを付与します。

### 評価手法

**Non-Cross-Video Setting**

- **同一動画を用いた学習・テスト**
- DEAP データセットに対して **5-fold Cross-Validation** を実施
- ベンチマークとして高い分類精度を確認

**Cross-Video Setting（LOVO）**

- **Leave-One-Video-Out（LOVO）戦略**を採用
- 学習とテストで **異なる動画刺激**を使用
- 各実験では、動画評価スコアに基づき
    - 上位 50% の動画
    - 下位 50% の動画
        
        から学習用動画を選択
        

### 要因分析

- **EEG 周波数帯の影響**
    - 単一周波数帯（θ / α / β / γ）のみを入力として LOVO 実験を実施
    - Arousal では **α・β 帯**、Valence では **β・γ 帯**が有効であることを確認
- **感情ラベルの信頼性**
    - **Rank-based ラベル**（評価スコア順位に基づく）
    - **Random ラベル**（ランダム割当）の 2 手法を比較
    - DEAP ではラベル信頼性が高く、AMIGOS / DREAMER ではラベルノイズの影響が示唆されます

## **パラメーター**設定

本プロジェクトの**主要な設定項目はすべて `config.py` に定義されています**。

以下では、実験の再現および調整に重要なパラメータについて説明します。

### 基本グローバル設定

- **`random_seed`**：乱数シード（デフォルト：42）。実験結果の再現性を確保するために使用します。
- **`window_size`**：信号処理に用いる時間窓のサイズ。
    - `64`：0.5 秒に相当
    - `1280`：10 秒に相当
    - `2560`：20 秒に相当
- **`overlap`**：スライディングウィンドウにおける重複サンプル数（現在は 0）。

### モデル学習ハイパーパラメータ

深層学習モデルの学習挙動を制御するパラメータです。

- **`epoch`**：学習エポック数（デフォルト：100）。
- **`batch_size`**：バッチサイズ（デフォルト：64）。
- **`learning_rate`**：初期学習率（デフォルト：0.001）。
- **`dropout_rate`**：ドロップアウト率（デフォルト：0.2）。過学習の抑制に使用します。
- **`filters`**：畳み込み層におけるフィルタ数（例：`[64, 128, 256]`）。
- **`kernel_size_list`**：畳み込みカーネルサイズのリスト。

### データセットパス設定（Dataset Paths）

本プロジェクトは **DEAP・AMIGOS・DREAMER** の 3 つの EEG データセットに対応しています。

主なパス設定は以下の通りです。

- **前処理済みデータ**：各データセットの `processed_data` パス
- **モデル専用データ**：CNN-Transformer 用の 10 秒ウィンドウ EEG データパス
- **周波数帯特徴量**：θ / α / β / γ 各帯域の特徴量データパス

### データセットメタ情報

- **ラベル情報**：各動画に対応する Valence・Arousal の評価値および二値ラベル定義
- **動画分割設定**：高／低 Valence・Arousal に基づく動画 ID リストを事前定義し、
    
    Cross-Video 実験や特定条件下の評価に使用します。
    

**チャネル対応（Channel Mapping）**

- DREAMER / AMIGOS では、14 チャネル EEG（例：AF3、F7、T7 など）を統一的に使用
- チャネル番号と電極名の対応関係を事前に定義しています。

### 補助機能

- **`random_video_list`**：乱数シードに基づき動画リストをシャッフルし、
    
    学習用／評価用に分割するための補助関数です。
    

> Note
> 
> 
> データセットのパス変更や実験条件の調整（例：ウィンドウサイズ、学習率の変更）を行う場合は、
> 
> `config.py` 内の該当変数を直接編集してください。
> 

# 研究背景


## 情動識別：価数 & 覚醒度 (Valence & Arousal)

![image.png](image\image6.png)

**情動識別（Emotion Recognition）**とは，映画・音楽・ゲームなどの刺激に対して人が示す生理信号（EEG，ECG，EOG，呼吸，GSR など）を解析し，深層学習モデルを用いて感情状態を自動的に推定する手法である。本研究では，**EEG 信号を入力**として用い，感情に関連する**時系列的および空間的特徴**を学習する。さらに，**Valence（快‐不快）および Arousal（覚醒度） といった感情次元空間上で感情をモデル化・予測することで，人間の感情反応を客観的に定量化・識別**することを目的とする。

## クロスビデオ情動識別（英訳：Cross-Video Emotion Recognition）とは？

![image.png](image\image7.png)

本研究が定義する「クロスビデオ情動識別」とは、モデルが訓練時に一部のビデオ刺激（video stimuli）下で収集された生理信号（EEG/ECG等）のみを学習し、テスト時には未学習の新しいビデオによって誘発された情動状態を予測することを指します。

換言すれば、訓練データとテストデータが「ビデオの次元」において重複しないことを意味します。その目的は、ビデオの内容・リズム・刺激強度に由来する固定的なパターンを記憶することではなく、「感情そのものの生理学的表象」を学習することにあります。

DEAP、AMIGOS、DREAMERなどの公開データセットにおいて、感情ラベルは通常、被験者が各ビデオを視聴した後の評価（Valence/Arousal等）に基づいています。しかし、ビデオごとの内容、テンポ、音量、映像変化の差異により、生理反応の統計的特性は大きく変化します。そのため、モデルは「精度は高く見えるが、実際には特定のビデオ刺激に過学習している」という現象に陥りやすくなります。

同一ビデオ内で訓練とテストを行う（Intra-video）設定では高い性能を示しても、未知のビデオ（Cross-video）に切り替わると性能が著しく低下するという課題に対し、本プロジェクトはロバストな識別手法を提供します。

## 留一交差検証 (Leave-One-Video-Out, LOVO)

![image.png](image\image8.png)

**Leave-One-Video-Out（LOVO）** は、モデルの**クロスビデオ汎化性能**を評価するための検証手法です。

図に示すように、各実験では **1 本の動画をテスト用**として保持し、残りの動画を学習用として使用します。この操作を、データセット内のすべての動画について順番に繰り返します。

本研究では、情動強度の偏りを抑えるため、動画の評価スコア（Valence / Arousal）に基づく **Top / Bottom 50% 動画戦略**を採用します。

- 評価スコアが上位 50% の動画から 10 本を選択
- 評価スコアが下位 50% の動画から 10 本を選択
- 合計 **20 本の動画を学習データ**として使用し、テスト動画は常に学習から除外します

この LOVO 戦略により、モデルが特定の動画内容に過度に依存することを防ぎ、**未知の動画刺激に対する情動認識性能**をより厳密に評価することが可能になります。これは、Cross-Video Emotion Recognition において不可欠な評価手法です。

# **開発環境とプロジェクト構成**

## **技術スタック**

- **使用言語・実行環境**
    
    Python
    
    （Conda / venv による仮想環境管理を推奨）
    
- **深層学習フレームワーク**
    - **PyTorch**：CNN-Transformer ベースのモデル実装
    - **TensorFlow / Keras**：2D CNN モデルの実装および比較実験
- **数値計算・データ前処理**
    
    NumPy，SciPy
    
- **実験管理・ログ記録**
    - TensorBoard（学習ログ・イベントファイル）
    - CSV 形式による学習曲線・評価結果の保存
- **結果出力・可視化補助**
    
    openpyxl（実験結果の Excel 出力）
    
- **開発・解析支援**
    
    Jupyter Notebook（データ解析・可視化・補助実験）

# データ準備

## データセット

本研究では，DEAP，AMIGOS，DREAMER データセットに含まれる EEG データを使用します。各データセットのダウンロードリンクは以下に示します。

- [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
- [AMIGOS](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/download.html)
- [DREAMER](https://www.kaggle.com/code/yousefgomaa43/dreamer-eeg)

## データ処理フロー

### 共通参照に平均化

![image.png](image\image9.png)

DREAMER データセットは，他の 2 つのデータセットと比較して共通参照への平均化が行われていません。そのため，データセット間でデータ処理フローを統一する目的で，本研究では DREAMER データセットに対して共通参照平均化を手動で実施し，併せて外れ値の除去を行います。

```python
def average_reference(video_eeg_stimuli):
    """
    Common Average Referencing (CAR)
    """
    mean_ref = np.mean(video_eeg_stimuli, axis=1, keepdims=True)
    video_eeg_stimuli_ref = video_eeg_stimuli - mean_ref
    return video_eeg_stimuli_ref

def remove_outliers_rolling(eeg_data, window_size=config.window_size, threshold=3):
    """
    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data with shape (time, channel)
    window_size : int
        Size of the sliding window
    threshold : float
        Threshold for outlier detection (number of standard deviations)
    """
    eeg_cleaned = np.copy(eeg_data)

    for ch in range(eeg_data.shape[1]):  # iterate over each channel
        series = pd.Series(eeg_data[:, ch])

        # Compute rolling mean and rolling standard deviation
        # center=True indicates that the current point is centered in the window
        rolling_mean = series.rolling(window=window_size, center=True).mean()
        rolling_std = series.rolling(window=window_size, center=True).std()

        # Identify outliers: deviation exceeds threshold × std
        outliers = np.abs(series - rolling_mean) > (threshold * rolling_std)
        # Replace outliers with NaN
        series[outliers] = np.nan
        # Interpolate NaN values (linear interpolation by default)
        series = series.interpolate(method='linear', limit_direction='both')
        # Write back the cleaned channel data
        eeg_cleaned[:, ch] = series.to_numpy()

        plt.plot(eeg_data[:, 0], label='Original')
        plt.plot(eeg_cleaned[:, 0], label='After Cleaning')
        plt.legend()
        plt.title("Comparison of Original and Cleaned Data")
        plt.show()

        print(f"Channel {ch} completed, number of outliers removed: {np.sum(outliers)}")

    return eeg_cleaned
```

### EEG 信号のフィルタリング

2D-CNN モデルの入力データとして用いる EEG 信号には，一連の前処理が必要です。まず，EEG 信号に対してフィルタリングを行い，異なる周波数帯に対応する 4 種類の脳波に分離します。異なる種類の脳波は，感情認識においてそれぞれ異なる役割を果たす可能性があります。

本研究では，**バターワース（Butterworth）フィルタ**を用いて EEG 信号のフィルタリングを行います。

```python
from scipy.signal import butter, lfilter

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # Compute the Nyquist frequency
    nyq = 0.5 * fs
    # Normalize the cutoff frequencies
    low = lowcut / nyq
    high = highcut / nyq
    # Design a Butterworth band-pass filter
    b, a = butter(order, [low, high], btype='band')
    # Apply the filter to the signal
    y = lfilter(b, a, data)

    return y
```

本研究で使用する各周波数帯およびその概要を以下に示します。

- **θ（Theta）波：4–8 Hz**
    
    低覚醒状態やリラックス状態，内省的な認知活動と関連するとされており，感情の安定性や注意状態を反映する可能性があります。
    
- **α（Alpha）波：8–13 Hz**
    
    安静時やリラックス状態で顕著に現れる周波数帯であり，感情の快・不快（Valence）との関連が報告されています。
    
- **β（Beta）波：13–30 Hz**
    
    覚醒度の高い状態や認知的負荷，緊張状態と関係しており，感情の興奮度（Arousal）を反映する可能性があります。
    
- **γ（Gamma）波：30–45 Hz**
    
    高次の認知処理や感情反応と関連するとされ，感情刺激に対する即時的・局所的な脳活動を捉える指標として用いられます。
    

### **特徴抽出**

2D-CNN モデルにデータを入力する前に，**微分エントロピー（Differential Entropy：DE）** および **パワースペクトル密度（Power Spectral Density：PSD）** を特徴量として抽出します。

- **微分エントロピー（DE）**
    
    EEG 信号の分布特性や情報量を表す指標であり，感情状態の変化を捉えるための特徴量として用いられます。
    
- **パワースペクトル密度（PSD）**
    
    各周波数帯における信号強度を表し，感情状態と関連する脳活動の特徴を反映します。
    

```python
from scipy.integrate import simpson as simps
from scipy.signal import welch

def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2

def compute_PSD(window_signal, low, high, fs=128):
    freqs, psd = welch(window_signal, fs=fs, nperseg=64, 
                             scaling='density', average='mean')
   
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    freq_res = freqs[1] - freqs[0]
    beta_power = simps(psd[idx_band], dx=freq_res)
    
    return beta_power
```

# 深層学習モデル

---

## 2D-CNN

![image.png](image\image10.png)

**2D-CNN モデル（2 次元畳み込みニューラルネットワーク）** は、EEG の**空間―周波数表現**から情動に関連する判別的特徴を学習するための手法である。

図に示すように、まず EEG 電極の頭皮上の配置関係に基づいて多チャネル信号を **2 次元トポグラフィグリッド** に変換し、さらに各周波数帯（θ / α / β / γ）ごとに 2D 特徴マップを構成して、チャネル方向にスタックすることで 2D 入力テンソルを生成する。

モデル内部では、複数層の **Conv2D（3×3 および 1×1 カーネル）** により局所的かつ高次の空間パターンを段階的に抽出し、**MaxPooling** によって特徴量の次元削減と計算効率の向上を行う。最終的に、特徴は **Flatten** と **全結合層（Dense）** に入力され、**Batch Normalization** と **Dropout** を組み合わせることで学習の安定性と汎化性能を高め、情動状態の推定を行う。

## CNN-Transformer

![image.png](image\image11.png)

**CNN-Transformer モデル**は、1 次元 CNN と Transformer Encoder を組み合わせ、**生の EEG 脳波時系列データ**から情動に関する特徴を直接学習するためのモデルです。

本モデルの目的は、**深層学習モデルが手工的な特徴設計に依存せず、原始脳波信号から情動特徴を自動的に学習できるかを検証すること**です。

図に示すように、まず **1D-CNN** を用いて異なる時間スケールの局所的な時系列パターンを抽出します。次に、**CLS トークン**と**位置エンコーディング**を付加し、**Transformer Encoder** によって長距離の時間依存関係をモデル化します。最後に全結合層を通して、情動状態の分類または回帰を行います。

本モデルは、CNN の局所的特徴抽出能力と Transformer のグローバルな時系列モデリング能力を統合し、EEG 情動認識におけるエンドツーエンド学習を実現します。

## **Fine-Tuning Pre-Trained Model**

![image.png](image\image12.png)

事前学習モデルのファインチューニング（Fine-tuning）は、転移学習を用いてモデルの性能および汎化能力を向上させる手法です。本研究では、大規模 EEG データを用いて事前学習されたモデルを下流タスクに適用します。

本研究では、[EEGPT](https://github.com/BINE022/EEGPT) を事前学習モデルとして採用します。を事前学習モデルとして採用します。EEGPT は、**Transformer アーキテクチャに基づく自己教師あり（無監督）学習**によって事前学習された EEG 表現モデルです。事前学習段階では情動ラベルを用いず、脳波信号そのものから**汎用的な EEG 特徴表現**を獲得することを目的としています。下流の情動認識タスクでは、**線形プロービング（Linear Probing）**によるファインチューニングを採用します。具体的には、事前学習済みモデルの一部（バックボーン）を固定し、分類用の線形層のみを学習します。この設定により、EEGPT が獲得した汎用 EEG 表現を保持したまま、下流タスクへの適応性能および **Cross-Video Emotion Recognition** における有効性を検証します。

# Citation / Paper

本リポジトリを研究・論文・学術発表において利用される場合は、以下の関連論文をご引用ください。本コードは、EEG に基づく **Cross-Video Emotion Recognition** の実験設定および評価手法を再現・拡張することを目的としています。

### 📘 JSAI 2024

**A Comparative Study of Content-Dependent and Content-Independent Emotion Recognition Using Convolutional Neural Network Based on DEAP Dataset**

Zhiying Huang, A. Guo, Jianhua Ma

*Proceedings of the 38th Annual Conference of the Japanese Society for Artificial Intelligence (JSAI 2024)*

🔗 [https://www.jstage.jst.go.jp/article/pjsai/JSAI2024/0/JSAI2024_4Q3IS2d02/_article/-char/ja/](https://www.jstage.jst.go.jp/article/pjsai/JSAI2024/0/JSAI2024_4Q3IS2d02/_article/-char/ja/)

```
@inproceedings{huang2024jsai,
  title     = {A Comparative Study of Content-Dependent and Content-Independent Emotion Recognition Using Convolutional Neural Network Based on DEAP Dataset},
  author    = {Huang, Zhiying and Guo, A. and Ma, Jianhua},
  booktitle = {Proceedings of the 38th Annual Conference of the Japanese Society for Artificial Intelligence},
  year      = {2024},
  publisher = {Japanese Society for Artificial Intelligence}
}

```

---

### 📗 IEEE PICom 2024

**Subject-General and Subject-Specific Emotion Recognition Across Video Stimuli Using EEG Signals**

Zhiying Huang, A. Guo, Jianhua Ma

*Proceedings of the IEEE International Conference on Pervasive and Intelligent Computing (PICom 2024)*

🔗 [https://ieeexplore.ieee.org/abstract/document/10795396](https://ieeexplore.ieee.org/abstract/document/10795396)

```
@inproceedings{huang2024picom,
  title     = {Subject-General and Subject-Specific Emotion Recognition Across Video Stimuli Using EEG Signals},
  author    = {Huang, Zhiying and Guo, A. and Ma, Jianhua},
  booktitle = {Proceedings of the IEEE International Conference on Pervasive and Intelligent Computing},
  year      = {2024},
  publisher = {IEEE}
}
```