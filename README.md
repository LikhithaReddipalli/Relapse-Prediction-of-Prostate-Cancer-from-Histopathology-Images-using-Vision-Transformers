Computational pathology, an emerging field in pathology, uses advancements in tissue
scanning and computer vision to automate disease analysis, particularly in cancer tissues.
This thesis centres on prostate cancer, the most prevalent non-skin cancer in men,
with diagnosis relying primarily on tissue samples graded by pathologists using the
Gleason score method. The reliability of diagnosis is, however, hampered by high
interobserver variability in Gleason scores. Thus, this thesis makes a transition towards
automatic cancer relapse prediction to identify whether a patient has a relapse within
five years after the radical prostatectomy, especially using vision transformers (ViT).
The primary topics investigated are how well vision transformers predict relapse using
internal pathology data and how performance may be enhanced by adding domainspecific
data pre-training from the publicly available Prostate cANcer graDe Assessment
Challenge dataset (PANDA). The research also explores the effects of utilising several
image resolutions in histopathology images with hierarchical vision transformers for
analysis of histopathology images, with the goal of capturing information at various image
scales for improved prediction accuracy. The findings spotlight a superior performance
of vision transformers coupled with multiple instance learning (MIL) over convolutional
neural networks with MIL, especially on the in-house data regarding relapse classification.
The results show that ViT combined with MIL outperforms CNN with MIL when
applied to the the internal data for relapse classification. However, the pre-training on
PANDA dataset did not significantly improve the relapse prediction performance despite
considerable hyperparameter testing and the use of multiple instance learning with
different pooling strategies. The resulting performance of hierarchical vision transformers
did not increase substantially over ViT with the MIL model, implying that hierarchical
vision transformers may not be inherently appropriate for this task, given the amount
of data. This study highlights the potential and pitfalls of using vision transformers in
computational pathology, specifically in prostate cancer recurrence prediction, offering
information on a path towards more precise prognostic models.


Keywords: Prostate cancer, multiple instance learning, vision transformers, hierarchical
vision transformers, pre-training
