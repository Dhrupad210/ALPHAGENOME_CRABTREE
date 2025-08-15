##alphagen4feature
A reproducible pipeline that processes 2,048-nt DNA windows from Crabtree-positive and Crabtree-negative yeasts into multi-omics features using AlphaGenome, then trains and tunes a suite of machine-learning classifiers for binary sequence classification.

##Why This Matters
Crabtree-positive yeasts (e.g., S. cerevisiae) preferentially ferment glucose even in the presence of oxygen, whereas Crabtree-negative species rely on respiration. These metabolic differences reflect underlying distinct regulatory programs, which should manifest in genomic signals like CAGE TSS activity, chromatin accessibility, histone modifications, and RNA output. The pipeline offers a way to computationally capture these differences for predictive modeling.

###What the Code Does

##Load data

##Reads FASTA files from crabtree_positive/ and crabtree_negative/ (60–70 sequences total).
Each sequence is padded or trimmed to 2,048 nt centered around its midpoint.


##Predict regulatory tracks with AlphaGenome

##Uses alphagenome.models.dna_client to generate per-position predictions for:

RNA-seq (transcriptional output proxy)
CAGE (TSS and enhancer-RNA initiation)
DNase-seq (chromatin accessibility / TF occupancy proxy)
Histone ChIP (promoter/enhancer state)


##Input is scoped using specified ontology term(s), e.g. UBERON:0000955.


##AlphaGenome can analyze sequences up to 1 million base pairs and predict thousands of functional genomic tracks at single-base resolution.


##Engineer features from tracks

##For each modality and strand, compute:

Sum, Max, Std, Ratio (pos/neg), Sum difference, Argmax position.


##Concatenate these across modalities → one feature vector per sequence.


##Aggregating base-level genomic signals (e.g., through sums, maxima, positional statistics) into fixed-length feature vectors is a standard strategy in both classical ML and deep models like DeepSEA, Enformer for region-level interpretation.


##Train + evaluate multiple models

##Models: Logistic Regression, Decision Tree, Random Forest, SVC, KNN, Gradient Boosting, XGBoost.
Preprocessing: feature standardization, stratified 90/10 train/test split.
Metrics: Accuracy, Macro-Precision, Recall, F1; confusion matrices saved in alphagenome_visualizations/ML_Results/.


##Hyperparameter tuning

Uses GridSearchCV with RepeatedStratifiedKFold (3×3) optimizing Macro-F1.
Reports best parameters, CV mean and standard deviation, and test set performance. Identifies the overall best model based on CV F1.




###Why Feature Selection is So Important
In genomics, data is high-dimensional and often sample-limited. Feature engineering—transforming dense tracks into interpretable, compact features—is both practical and biologically meaningful. This step is central to the pipeline's success, as it directly addresses key challenges in genomic modeling:

##Dimensionality Reduction:## Raw AlphaGenome predictions generate thousands of per-base values per track, leading to massive feature spaces. By selecting and summarizing into fixed-length vectors (e.g., sum, max, std), we drastically reduce dimensions while preserving essential signal patterns, preventing overfitting and enabling efficient training on small datasets like our ~64 sequences.
##Enhanced Interpretability:## Each engineered feature (e.g., strand ratio for directionality, argmax position for spatial context) has clear biological ties, allowing us to trace model decisions back to regulatory mechanisms. This transparency is vital for understanding Crabtree-related differences, far beyond black-box approaches.
##Improved Model Performance and Robustness:## Thoughtful feature selection captures the most discriminative aspects of genomic signals—total activity (sum), peak events (max), variability (std), asymmetry (ratio/diff), and location (position). This boosts signal-to-noise ratio, leading to higher accuracy and generalization. In classical ML, such curated features often outperform raw inputs, especially with limited samples.
##Biological Relevance:## Features are designed to highlight regulatory hallmarks, like unidirectional transcription or focal enhancers, making the model more aligned with yeast metabolic biology.

Overall, feature selection isn't just a preprocessing step—it's the pipeline's core strength, bridging raw predictions to actionable insights and enabling reliable classification despite data constraints.

##Why These Modalities Matter

##CAGE:## Identifies transcription start sites and bidirectional enhancer RNA; used in FANTOM promoter/enhancer mapping.
##DNase-seq:## Measures open chromatin and TF occupancy; a cell-type-specific regulatory hallmark.
##Histone ChIP:## Marks like H3K4me3 (promoters), H3K27ac/H3K4me1 (enhancers) distinguish regulatory state.
##RNA-seq:## Represents transcriptional output, summarizing net regulatory effect.

These modalities provide complementary insights into transcription, chromatin architecture, and regulatory dynamics—essential for modeling metabolic state differences.

##Feature Design Rationales
Below are the six key features engineered from per-strand functional genomic track predictions—each with logic and justification, emphasizing their role in capturing biologically meaningful patterns.
##1. Sum of Signal (_sum_pos, _sum_neg)

Logic:

Represents the total predicted activity within the 2 kb window for a given track and DNA strand. For RNA-seq, it serves as a proxy for cumulative gene expression; for DNase-seq or histone ChIP, it reflects overall chromatin accessibility or binding intensity.
Justification:

A higher total sum of RNA-seq signals, for instance, might be a strong indicator of a highly expressed gene in that region, a key factor in the metabolic state of the yeast. This is a fundamental measure of overall activity.

##2. Maximum Signal (_max_pos, _max_neg)

Logic:

Identifies the peak value—the single highest signal point—in the 2 kb window. Example: a CAGE signal spike marking a dominant TSS, or a strong DNase peak indicating a key regulatory site.
Justification:

Biological systems are often driven by dominant regulatory elements. The presence of a strong promoter (high CAGE peak) or a single, highly active regulatory site (high ChIP-histone peak) is often more informative than the average activity across the entire sequence. This feature captures the magnitude of these critical events.

##3. Signal Variability (Standard Deviation, _std_pos, _std_neg)

Logic:

Measures how variable or "spiky" the signal is across the window. High standard deviation suggests uneven, peak-heavy regions, while low values imply uniform signal distribution.
Justification:

The spatial pattern of regulatory elements is a key part of gene regulation. A sequence with a high standard deviation for DNase-seq might indicate a few specific, highly accessible regulatory sites surrounded by inaccessible chromatin. In contrast, a low standard deviation might suggest a more uniformly open or closed chromatin state. This feature helps distinguish between these different regulatory architectures.

##4. Strand Ratio (_ratio_pos_neg)

Logic:

Computes the ratio of total positive-strand signal to negative-strand signal. Ratios significantly above 1 indicate stronger positive-strand activity, suggesting unidirectional transcription or strand-specific regulation.
Justification:

The relative balance of activity between the two strands is a powerful indicator of transcription and regulatory directionality. For a gene, a high positive-to-negative strand ratio in RNA-seq is expected. Deviations from this pattern might signal antisense transcription or complex regulatory overlaps, which could be important biological markers.

##5. Difference in Sum (_diff_sum)

Logic:

Provides the absolute difference between positive and negative strand total signals—i.e., sum_pos – sum_neg.
Justification:

This feature offers a linear and magnitude-based measure of strand imbalance, less affected by very small denominators than ratios. Large positive differences reflect strong unidirectional activity; near-zero values may indicate no transcription or bidirectional regulation.

##6. Position of Maximum (_pos_max_pos, _pos_max_neg)

Logic:

Records the location (index) of the highest signal within the 2 kb window for each strand.
Justification:

The location of regulatory elements relative to a gene's start or end site is often crucial. For example, the proximity of a predicted CAGE peak (TSS) to the beginning of a gene is a fundamental characteristic. Furthermore, the model may be detecting regulatory elements like enhancers or silencers, and their position relative to the main gene body can be a critical determinant of their function. This feature allows the model to learn patterns related to spatial organization.


###Quick Start
Install necessary packages:
bashpip install alphagenome biopython xgboost scikit-learn pandas numpy
Refer to the AlphaGenome docs for environment setup.
##Set your AlphaGenome API key:
bashexport ALPHAGENOME_API_KEY="YOUR_KEY_HERE"
##Prepare data:
Place FASTA files in:

crabtree_positive/*.fa
crabtree_negative/*.fa

##Run the notebook:

Loads and preprocesses sequences to 2,048 nt.
Predicts signals via AlphaGenome.
Engineers features and concatenates per sequence.
Trains baseline models and evaluates performance.
Conducts hyperparameter tuning and saves results.

##Outputs

Feature table: Aggregated metrics per sequence per modality per strand.
Model performance: Metrics (Accuracy, Macro Precision/Recall/F1) for all models.
Best tuned model: Summary based on cross-validation F-score.
Visualization: Confusion matrices (PNG files) for interpretability.

##Notes & Limitations

##Cross-species applicability:## AlphaGenome is trained on human/mouse data; applying it to yeast introduces domain-shift. Interpret with caution.
##Window length:## The 2 kb window centers promoter context but excludes long-range elements (e.g., distal enhancers >10 kb).
##Feature-engineering trade-off:## Manual summaries are interpretable and lightweight, but may underperform end-to-end deep models modeling base-level interactions.
