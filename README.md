AGFormer is a graph-based spatiotemporal forecasting framework for multi-reservoir inflow prediction. It integrates an adaptive graph learning module and a Transformer-based encoder–decoder to capture spatial dependencies and temporal dynamics.

Model Architecture
1. (Optional) Pretraining: A semi-supervised framework learns shared temporal representations from heterogeneous and misaligned historical records, allowing the model to leverage all available data before downstream forecasting.
2. Feature Extraction: Projects historical observations into latent embeddings via a shared MLP feature extractor.
3. Adaptive Graph Learning: Dynamically refines reservoir connections to enhance spatial information flow.
4. Temporal Encoding: A Transformer encoder–decoder models temporal dependencies.
5. Forecasting: Decodes latent representations into multi-day inflow predictions.
