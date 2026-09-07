# thesis-ad
# Tralzformer
Longitudinal Modeling of Alzheimer's Disease Progression and Diagnosis 

This repository contains code and experiments for my thesis on Alzheimer’s disease progression modeling using Transformer-based architectures.

The project focuses on two complementary prediction tasks:

thesis-ad-diagnosis/
Multi-class diagnosis prediction, i.e. predicting the subject’s clinical diagnosis (CN, MCI, AD) at each time step.

thesis-ad-conversion/
Next-visit conversion prediction, i.e. estimating whether a subject without AD at baseline will progress to AD at the following visit(s).

# Structure

├── thesis-ad-diagnosis/    # Preprocessing for diagnosis cohort and tralzformer experiment for diagnosis task
├── thesis-ad-conversion/   # Preprocessing for conversion cohort and tralzformer experiment for conversion task

# Versions

11/09/2025: v0.3.5: Final Experiments & Interpretability for all tasks.

11/09/2025: v0.3: Final Experiments & Interpretability. Missing AD vs Rest on Subgroups.

09/09/2025: v0.2: Final working code for current-visit diagnosis prediction. Conversion prediction needs inspection.

