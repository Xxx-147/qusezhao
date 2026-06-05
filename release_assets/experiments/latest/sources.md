# Sources

- Local user-supplied true negative/positive pairs.
- Wikimedia Commons Public Domain: Negative_Positive-Picture.jpg.
- BlueNeg raw DNG subset from Hugging Face: https://huggingface.co/datasets/ttgroup/blueneg-release
- BlueNeg paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_BlueNeg_A_35mm_Negative_Film_Dataset_for_Restoring_Channel-Heterogeneous_Deterioration_ICCV_2025_paper.pdf

Scanner notes:

- Fujifilm Frontier SP-3000 and Noritsu scanners are important target styles, but public paired raw-negative + SP3000/Noritsu target datasets were not found in a clearly reusable form.
- Future dataset rows should set `scanner_profile` to `Frontier SP3000`, `Noritsu`, or the exact lab/scanner profile whenever known.